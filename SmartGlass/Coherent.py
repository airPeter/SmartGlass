'''
    unit [um].
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .Coherent_model import optical_model
from .data_utils import create_circular_detector, input_label_process, draw_circular_detector, SimpleDataset, input_label_process_complex, CM
from .optical_utils import propagator, init_aperture
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import cv2

def toint(x):
    if type(x) == np.ndarray:
        return np.round(x).astype(int)
    else:
        return int(round(x))

class Substrate:
    def __init__(self, n, thickness) -> None:
        self.n = n
        self.thickness = thickness

class MNISTObj:
    def __init__(self, font, size) -> None:
        self.font = font
        self.size = size

class CirleDetector:
    def __init__(self, radius, coos) -> None:
        '''
        input:
            coos: a numpy array with shape (num_detectors, 2) store the coordinates for each cirular detector.
        '''
        self.radius = radius
        self.coos = coos
    def create(self, plane_size, step_size):
        r = toint(self.radius/step_size)
        Ks = toint(self.coos/step_size)
        return create_circular_detector(r, Ks, plane_size)
    def draw(self, img, step_size):
        r = toint(self.radius/step_size)
        Ks = toint(self.coos/step_size)
        return draw_circular_detector(img, r, Ks)
    def invalid(self,):
        '''
            check whether there's overlap between detectors.
        '''
        def overlap(ps, r):
            if len(ps) == 1:
                return False
            else:
                for i in range(1, len(ps)):
                    dis = np.sqrt(((ps[i] - ps[0])**2).sum())
                    if dis < r:
                        return True
                return overlap(ps[1:], r)
        return overlap(self.coos, self.radius)
class Coherent:
    def __init__(self, wavelength, num_layers, size, res, prop_dis, object, num_classes, substrate = None, aperture = False) -> None:
        self.wavelength = wavelength
        self.num_layers = num_layers
        if num_layers > 1:
            raise Exception("currently only support one layer.")
        self.size = size
        self.res = res
        self.step_size = 1/res
        self.plane_size = toint(self.size/self.step_size)
        self.prop_dis = prop_dis
        self.object = object
        self.num_class = num_classes
        self.detector = None
        self.substrate = substrate
        self.aperture = aperture
        self.model = None
        self.dtype = torch.float32
        self.cdtype = torch.complex64
        self.device = None
        self.out_path = None
        
    def init_model(self, device, init_phase = None, init_detector = None):
        self.device = device
        prop = propagator(self.plane_size, self.step_size, self.prop_dis, self.wavelength)
        f_kernel = np.fft.fft2(np.fft.ifftshift(prop))
        f_kernel = torch.tensor(f_kernel, device='cpu', dtype=self.cdtype, requires_grad = False)
        f_kernel_sub = None
        if not (self.substrate is None):
            prop = propagator(self.plane_size, self.step_size, self.substrate.thickness, self.wavelength/self.substrate.n)
            f_kernel_sub = np.fft.fft2(np.fft.ifftshift(prop))
            f_kernel_sub = torch.tensor(f_kernel_sub, device='cpu', dtype=self.cdtype, requires_grad = False)
        aperture = None
        if self.aperture:
            aperture = init_aperture(self.plane_size)
            aperture = torch.tensor(aperture, device='cpu',dtype=self.dtype, requires_grad = False)
        self.model = optical_model(self.num_layers, self.plane_size, f_kernel, f_kernel_sub, aperture, self.num_class)
        self.init_paras(init_phase, init_detector)
        self.model.to(device)
    
    def assign_detector(self, detector):
        self.detector = detector
        circular_mask = detector.create(self.plane_size, self.step_size)
        circular_mask = torch.tensor(circular_mask, dtype=self.dtype, requires_grad = False)
        state_dict = self.model.state_dict()
        state_dict['mask_const'] = circular_mask
        self.model.load_state_dict(state_dict)
        print("detector assigned.")
        
    def init_paras(self, init_phase, init_detector):
        self.model.reset()
        if init_phase is None:
            print('initialized by default phase paras.')
        else:
            self.assign_phase(init_phase)
        self.assign_detector(init_detector)
            
    def assign_phase(self, init_phase):
        init_phase = np.reshape(init_phase, (1, 1, self.plane_size, self.plane_size))
        init_phase = torch.tensor(init_phase, dtype = self.dtype)
        state_dict = self.model.state_dict()
        state_dict['optical.phase'] = init_phase
        self.model.load_state_dict(state_dict)
        print("phase assigned.")
        return None
    def gen_obj(self, img, random_shift = False, background = False):
        '''
            process a img into a model input.
        '''
        if img.dtype == complex:
            if img.shape[0] < self.plane_size:
                x_abs = np.abs(img)
                x_ph = np.angle(img)
                abs_resized = cv2.resize(x_abs,(self.plane_size, self.plane_size))
                ph_resized = cv2.resize(x_ph,(self.plane_size, self.plane_size))
                img = abs_resized * np.exp(1j * ph_resized)
            img = np.reshape(img, (1, img.shape[0], img.shape[1]))
            return img
        object_size = toint(self.object.size / self.step_size)
        out_x, _ = input_label_process(img, None, self.plane_size, object_size, self.object.font, random_shift = random_shift, background = background)
        return out_x
    
    def gen_obj_complex(self, field, random_shift = False, background = False):
        '''
            process a img into a model input.
        '''
        object_size = toint(self.object.size / self.step_size)
        out_x, _ = input_label_process_complex(field, None, self.plane_size, object_size, random_shift = random_shift, background = background)
        return out_x
    def forward(self, obj):
        data_input = torch.as_tensor(obj)
        
        data_input = data_input.to(self.device, dtype = self.cdtype)
        with torch.no_grad():
            signal, logit = self.model(data_input, noise = None)
        signal = signal.cpu().numpy()
        return signal
    
    def optimze_detector(self, batch_size, data_path, select_idx, subspace_size = 10, n_iter = 20):
        from gradient_free_optimizers import HillClimbingOptimizer
        from gradient_free_optimizers import RandomRestartHillClimbingOptimizer
        from gradient_free_optimizers import EvolutionStrategyOptimizer
        radius = self.detector.radius
        init_coos = self.detector.coos
        def obj_function(paras):
            '''
            paras is a dict that store the position of each detector.
            '''
            coos = dict2arr(paras)
            tmp_detector = CirleDetector(radius, coos)
            if tmp_detector.invalid():
                return 0
            self.assign_detector(tmp_detector)
            score = self.test(batch_size, data_path, select_idx)
            return score 
        domain_min = 2 * radius
        domain_max = self.size - domain_min
        sub_space = np.round(np.linspace(domain_min, domain_max, subspace_size))
        print("sub space:")
        print(sub_space)
        search_space = {}
        for i in range(len(init_coos)):
                search_space['y' + str(i)] = sub_space.copy()
                search_space['x' + str(i)] = sub_space.copy()
        
        # init_paras = fit2space(sub_space, init_coos)
        # init_paras = arr2dict(init_paras)
        init_paras = arr2dict(init_coos)
        print("init paras:")
        print(init_paras)
        initialize={"warm_start": [init_paras]}
        #opt = HillClimbingOptimizer(search_space, initialize)
        opt = RandomRestartHillClimbingOptimizer(search_space, initialize, n_iter_restart=10)
        #opt = EvolutionStrategyOptimizer(search_space, initialize)
        early_stopping = {
            'n_iter_no_change': 50, #50 iter no change then stop.
            'tol_rel': 1 #increase at least 1 percent.
        }
        opt.search(obj_function, n_iter=n_iter, early_stopping= early_stopping)
        print(f"best acc: {opt.best_score}")
        print("If you want to continue to train the phase, remember to assign detector by the best paras.")
        best_coos = dict2arr(opt.best_para)
        best_detector = CirleDetector(radius, best_coos)
        return best_detector

    def optimze(self, lr, beta, batch_size, epoches, notes, mu_white_noise, data_path, init_coos,
                out_path = 'output_coherent/', subspace_size = 10, n_iter = 50, population = 1):
        from gradient_free_optimizers import RandomRestartHillClimbingOptimizer
        from gradient_free_optimizers import RandomSearchOptimizer
        from gradient_free_optimizers import ParticleSwarmOptimizer
        from gradient_free_optimizers import EvolutionStrategyOptimizer
        
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        self.out_path = out_path = out_path + notes + '/'
        x_train = np.load(data_path+'x_train_f.npy')
        y_train = np.load(data_path+'y_train_f.npy')    
        def trans2obj(sample):
            sample['X'] = self.gen_obj(sample['X'])
            return sample
        dataset_train = SimpleDataset(x_train, y_train, trans2obj)
        dataloader_train = DataLoader(dataset_train,
                                shuffle=True,
                                batch_size= batch_size)
        print(f"Number of batches per epoch: {len(dataloader_train)}.")
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        prop_noise_dist = torch.distributions.normal.Normal(0, 0.03)
        white_noise_dist = torch.distributions.normal.Normal(mu_white_noise, 0.6 * mu_white_noise)
        radius = self.detector.radius
        Iters = []
        def obj_function(paras):
            writer = SummaryWriter(out_path + 'summary_' + str(sum(Iters)))
            total_steps = epoches * len(dataloader_train)
            coos = dict2arr(paras)
            tmp_detector = CirleDetector(radius, coos)
            if tmp_detector.invalid():
                return 0
            self.model.reset()
            self.assign_detector(tmp_detector)
            
            for epoch in range(epoches):
                for step, sample in enumerate(dataloader_train):
                    X, Y = sample['X'], sample['Y']
                    X_input, Y_input = torch.as_tensor(X).to(self.device, dtype = self.cdtype), torch.as_tensor(Y).to(self.device)
                    prop_noise = prop_noise_dist.sample(torch.Size([1, self.plane_size, self.plane_size])).to(self.device, dtype = torch.double)
                    white_noise = white_noise_dist.sample(torch.Size([1, self.plane_size, self.plane_size])).to(self.device, dtype = torch.double)
                    noise = (prop_noise, white_noise)
                    signal, logit = self.model(X_input, noise)
                    # Calculate loss 
                    err1 = loss(logit, Y_input)
                    err2 = - beta * logit.sum()
                    err = err1 + err2
                    # Calculate gradients in backward pass
                    optimizer.zero_grad()
                    err.backward()
                    optimizer.step()
                    global_step = epoch * len(dataloader_train) + step

                    log_freq = int((len(dataloader_train))/10)
                    if log_freq == 0:
                        log_freq = 1
                    if (global_step + 1)% log_freq == 0:
                        writer.add_scalar('crossentropy loss',
                            scalar_value = err1.item(), global_step = global_step)
                        writer.add_scalar('logit sum loss',
                            scalar_value = err2.item(), global_step = global_step)
                        
            acc = self.test(batch_size, data_path)
            writer.add_scalar('Test accuracy',
                scalar_value = np.round(acc * 100, 1), global_step = global_step)
            obj_img = np.squeeze(X[0])
            writer.add_figure('obj',
                            get_img(obj_img, str(Y[0].item())),
                            global_step= global_step)
            I_img = signal.detach().cpu().numpy()[0]
            I_img = np.squeeze(I_img[0])
            I_img = self.detector.draw(I_img, self.step_size)
            writer.add_figure('I_img',
                            get_img(I_img, ''),
                            global_step= global_step)  
            phase = self.model.optical.phase.detach().cpu().numpy()
            phase = np.squeeze(phase)
            writer.add_figure('phase',
                            get_img(phase),
                            global_step= global_step) 
            Iters.append(1)
            return acc
        
        domain_min = 2 * radius
        domain_max = self.size - domain_min
        sub_space = np.round(np.linspace(domain_min, domain_max, subspace_size))
        print("sub space:")
        print(sub_space)
        search_space = {}
        for i in range(self.num_class):
                search_space['y' + str(i)] = sub_space.copy()
                search_space['x' + str(i)] = sub_space.copy()
        
        init_paras = [arr2dict(x) for x in init_coos] 
        print("init paras:")
        print(init_paras)
        initialize={"warm_start": init_paras}
        #opt = RandomSearchOptimizer(search_space, initialize)
        #opt = ParticleSwarmOptimizer(search_space, initialize, population = population)
        opt = EvolutionStrategyOptimizer(search_space, initialize, population = population)
        opt.search(obj_function, n_iter=n_iter)
        best_coos = dict2arr(opt.best_para)
        np.savetxt(out_path + "best_detector_coos.csv", best_coos, delimiter= ',')   
        opt.search_data.to_csv(out_path + "searched_detector_coos.csv")
        return None
    
    def optimze_optics(self, lr, beta, batch_size, epoches, test_freq, notes, mu_white_noise, data_path, out_path = 'output_coherent/'):
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        self.out_path = out_path = out_path + notes + '/'
        writer = SummaryWriter(out_path + 'summary1')
        x_train = np.load(data_path+'x_train_f.npy')
        y_train = np.load(data_path+'y_train_f.npy')    
        def trans2obj(sample):
            sample['X'] = self.gen_obj(sample['X'])
            return sample
        dataset_train = SimpleDataset(x_train, y_train, trans2obj)
        dataloader_train = DataLoader(dataset_train,
                                shuffle=True,
                                batch_size= batch_size)
        print(f"Number of batches per epoch: {len(dataloader_train)}.")
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        prop_noise_dist = torch.distributions.normal.Normal(0, 0.03)
        white_noise_dist = torch.distributions.normal.Normal(mu_white_noise, 0.6 * mu_white_noise)
        for epoch in tqdm(range(epoches)):
            for step, sample in enumerate(tqdm(dataloader_train, leave=False)):
                X, Y = sample['X'], sample['Y']
                X_input, Y_input = torch.as_tensor(X).to(self.device, dtype = self.cdtype), torch.as_tensor(Y).to(self.device,dtype = torch.long)
                prop_noise = prop_noise_dist.sample(torch.Size([1, self.plane_size, self.plane_size])).to(self.device, dtype = torch.double)
                white_noise = white_noise_dist.sample(torch.Size([1, self.plane_size, self.plane_size])).to(self.device, dtype = torch.double)
                noise = (prop_noise, white_noise)
                signal, logit = self.model(X_input, noise)
                # Calculate loss 
                err1 = loss(logit, Y_input)
                err2 = - beta * logit.sum()
                err = err1 + err2
                # Calculate gradients in backward pass
                optimizer.zero_grad()
                err.backward()
                optimizer.step()
                global_step = epoch * len(dataloader_train) + step
                final_step = (epoches - 1) * len(dataloader_train) - 1
                if (global_step + 1)% test_freq == 0 or global_step == final_step:
                    acc = self.test(batch_size, data_path)
                    writer.add_scalar('Test accuracy',
                        scalar_value = np.round(acc * 100, 1), global_step = global_step)
                    obj_img = np.squeeze(X[0])
                    
                    writer.add_figure('obj',
                                    get_img(obj_img, str(Y[0].item())),
                                    global_step= global_step)
                    I_img = signal.detach().cpu().numpy()[0]
                    I_img = np.squeeze(I_img[0])
                    I_img = self.detector.draw(I_img, self.step_size)
                    writer.add_figure('I_img',
                                    get_img(I_img, ''),
                                    global_step= global_step)  
                    phase = self.model.optical.phase.detach().cpu().numpy()
                    phase = np.squeeze(phase)
                    writer.add_figure('phase',
                                    get_img(phase),
                                    global_step= global_step) 
                
                log_freq = int((len(dataloader_train))/10)
                if log_freq == 0:
                    log_freq = 1
                if (global_step + 1)% log_freq == 0:
                    writer.add_scalar('crossentropy loss',
                        scalar_value = err1.item(), global_step = global_step)
                    writer.add_scalar('logit sum loss',
                        scalar_value = err2.item(), global_step = global_step)
        np.savetxt(out_path + "phase.csv", phase, delimiter= ',')    
        acc = self.test(batch_size, data_path, confusion_matrix = True)
        writer.add_scalar('Test accuracy',
            scalar_value = np.round(acc * 100, 1), global_step = global_step)
        
    def test(self, batch_size, data_path, select_idx = None, confusion_matrix = False):
        self.model.eval()
        Y_pred = []
        Y_label = []
        x_test = np.load(data_path+'x_test_f.npy')
        y_test = np.load(data_path+'y_test_f.npy')   
        if not (select_idx is None):
            x_test = np.take(x_test, select_idx, axis = 0)
            y_test = np.take(y_test, select_idx, axis = 0)
         
        def trans2obj(sample):
            sample['X'] = self.gen_obj(sample['X'])
            return sample
        dataset_test = SimpleDataset(x_test, y_test, trans2obj)
        dataloader_test = DataLoader(dataset_test,
                                shuffle=False,
                                batch_size= batch_size)
        with torch.no_grad():
            for sample in dataloader_test:
                    X, Y = sample['X'], sample['Y']
                    Y_label += Y.tolist()
                    X_input, Y_input = torch.as_tensor(X).to(self.device, dtype = self.cdtype), torch.as_tensor(Y).to(self.device)
                    _, logit = self.model(X_input, None)
                    pred = np.argmax(logit.cpu().numpy(), axis = -1)
                    Y_pred = Y_pred + pred.tolist()
        right_num = np.equal(Y_label, Y_pred).sum()
        test_acc = right_num/len(Y_label)
        if confusion_matrix:
            CM(Y_label, Y_pred, self.num_class, self.out_path)
        #print(f"Test accuracy: {test_acc * 100:.2f}%.")
        return test_acc

def get_img(img, title = None):
    if img.dtype == torch.complex128:
        fig, axes = plt.subplots(1, 2, figsize = (10, 4))
        plot0 = axes[0].imshow(np.abs(img))
        plt.colorbar(plot0, ax = axes[0])
        axes[0].set_title('amplitude')
        plot1 = axes[1].imshow(np.angle(img))
        plt.colorbar(plot1, ax = axes[1])
        axes[1].set_title('phase')
    else:
        fig = plt.figure()
        plt.imshow(img)
        plt.colorbar()
    if not (title is None):
        plt.title(title)
    return fig

def dict2arr(paras):
    coos = []
    for key in paras.keys():
        coos.append(paras[key])
    coos = np.array(coos)
    coos = coos.reshape(-1, 2)
    return coos
def arr2dict(coos):
    #coos shape: [number of detectors, 2]
    paras = {}
    for i in range(len(coos)):
        paras['y' + str(i)] = coos[i, 0]
        paras['x' + str(i)] = coos[i, 1]
    return paras
def fit2space(sub_space, init_paras):
    shape = init_paras.shape
    init_paras = init_paras.reshape(-1,)
    for i in range(len(init_paras)):
        pi = init_paras[i]
        idx = np.argmin(np.abs(pi - sub_space))
        init_paras[i] = sub_space[idx]
    init_paras = init_paras.reshape(shape)
    return init_paras