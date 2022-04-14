'''
    unit [um].
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .Coherent_model import optical_model
from .data_utils import create_circular_detector, input_label_process, draw_circular_detector, SimpleDataset
from .optical_utils import propagator, init_aperture
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

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
            coos: a numpy array with shape (2, num_detectors) store the coordinates for each cirular detector.
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
        draw_circular_detector(img, r, Ks)
        
class Coherent:
    def __init__(self, wavelength, num_layers, size, res, prop_dis, object, detector, substrate = None, aperture = False) -> None:
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
        self.detector = detector
        self.substrate = substrate
        self.aperture = aperture
        self.model = None
        self.dtype = torch.float32
        self.cdtype = torch.complex64
        self.device = None
        
    def init_model(self, device, init_phase):
        self.device = device
        circular_mask = self.detector.create(self.plane_size, self.step_size)
        circular_mask = torch.tensor(circular_mask, device = 'cpu', dtype=self.dtype, requires_grad = False)
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
        self.model = optical_model(self.num_layers, self.plane_size, f_kernel, f_kernel_sub, aperture, circular_mask)
        self.init_paras(init_phase)
        self.model.to(device)
        
    def init_paras(self, init_phase = None):
        self.model.reset()
        if init_phase is None:
            print('initialized by default phase paras.')
            return None
        else:
            init_phase = torch.tensor(init_phase, dtype = torch.float)
            state_dict = self.model.state_dict()
            state_dict['optical.phase'] = init_phase
            self.model.load_state_dict(state_dict)
            print('initialized by loaded init_phase.')
            return None 
        
    def gen_obj(self, img, random_shift = False, background = False):
        '''
            process a img into a model input.
        '''
        object_size = toint(self.object.size / self.step_size)
        out_x, _ = input_label_process(img, None, self.plane_size, object_size, self.object.font, random_shift = random_shift, background = background)
        return out_x
    
    def forward(self, obj):
        data_input = torch.as_tensor(obj)
        data_input = data_input.to(self.device, dtype = self.dtype)
        with torch.no_grad():
            signal, logit = self.model(data_input, noise = None)
        signal = signal.cpu().numpy()
        return signal
    def optimze_optics(self, lr, beta, batch_size, epoches, test_freq, notes, mu_white_noise, data_path):
        out_path = 'output_coherent/'
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        out_path = out_path + notes + '/'
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
            for step, sample in enumerate(tqdm(dataloader_train)):
                X, Y = sample['X'], sample['Y']
                X_input, Y_input = torch.as_tensor(X).to(self.device, dtype = self.dtype), torch.as_tensor(Y).to(self.device)
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
                
                if (global_step + 1)% test_freq == 0:
                    self.test(batch_size, data_path)
                    obj_img = np.squeeze(X[0])
                    writer.add_figure('obj',
                                    get_img(obj_img, str(Y[0].item())),
                                    global_step= global_step)
                    I_img = signal.detach().cpu().numpy()[0]
                    I_img = np.squeeze(I_img[0])
                    I_img = self.detector.draw(I_img, self.step_size)
                    writer.add_figure('I_img',
                                    get_img(I_img),
                                    global_step= global_step)  
                    phase = self.model.optical.phase.detach().cpu().numpy()
                    phase = np.squeeze(phase)
                    writer.add_figure('phase',
                                    get_img(phase),
                                    global_step= global_step) 
                
                log_freq = int((len(dataloader_train))/10)
                if (global_step + 1)% log_freq == 0:
                    writer.add_scalar('crossentropy loss',
                        scalar_value = err1.item(), global_step = global_step)
                    writer.add_scalar('logit sum loss',
                        scalar_value = err2.item(), global_step = global_step)
        np.savetxt(out_path + "phase.csv", phase, delimiter= ',')    

    def test(self, batch_size, data_path):
        self.model.eval()
        Y_pred = []
        Y_label = []
        x_test = np.load(data_path+'x_test_f.npy')
        y_test = np.load(data_path+'y_test_f.npy')    
        def trans2obj(sample):
            sample['X'] = self.gen_obj(sample['X'])
            return sample
        dataset_test = SimpleDataset(x_test, y_test, trans2obj)
        dataloader_test = DataLoader(dataset_test,
                                shuffle=True,
                                batch_size= batch_size)
        with torch.no_grad():
            for sample in dataloader_test:
                    X, Y = sample['X'], sample['Y']
                    Y_label += Y.tolist()
                    X_input, Y_input = torch.as_tensor(X).to(self.device, dtype = self.dtype), torch.as_tensor(Y).to(self.device)
                    _, logit = self.model(X_input, None)
                    pred = np.argmax(logit.cpu().numpy(), axis = -1)
                    Y_pred = Y_pred + pred.tolist()
        right_num = np.equal(Y_label, Y_pred).sum()
        test_acc = right_num/len(Y_label)
        print(f"Test accuracy: {test_acc * 100:.2f}%.")
        return None

def get_img(img, title = None):
    fig = plt.figure()
    plt.imshow(img)
    plt.colorbar()
    if not (title is None):
        plt.title(title)
    return fig