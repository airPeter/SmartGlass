'''
    unit [um].
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .Coherent_model import optical_model
from .data_utils import create_circular_detector, input_label_process
from .optical_utils import propagator, init_aperture

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
    #def optimze_optics(self, lr, beta, batch_size, epoches, test_freq):
    
        