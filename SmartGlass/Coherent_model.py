import torch
import torch.nn.functional as f
import torch.fft

def fourier_conv(signal: torch.Tensor, f_kernel: torch.Tensor) -> torch.Tensor:
    '''
        args:
        signal, kernel: complex tensor, assume the images are square. the last 2 dim of signal is the height, and width of images.
    '''
    s_size = signal.size()
    k_size = f_kernel.size()
    padding = (k_size[-1] - s_size[-1])//2
    if (k_size[-1] - s_size[-1])%2== 0:
        signal = f.pad(signal, (padding, padding, padding, padding))
    else:
        signal = f.pad(signal, (padding, padding + 1, padding, padding + 1))

    f_signal = torch.fft.fftn(signal, dim = (-2, -1))

    f_output = f_signal * f_kernel
    f_output = torch.fft.ifftn(f_output, dim = (-2, -1))
    f_output = f_output[:,:,padding: padding + s_size[-1], padding:padding + s_size[-1]]
    
    return f_output

class optical_layer(torch.nn.Module):
    def __init__(self, plane_size):
        '''

        '''
        super(optical_layer, self).__init__()
        self.plane_size = plane_size
        self.phase = torch.nn.Parameter(torch.empty((1, 1, plane_size, plane_size)))

    def forward(self, signal):
        '''
            f: torch.nn.functional
        '''
        phase_real = torch.cos(self.phase)
        phase_imag = torch.sin(self.phase)
        c_phase = torch.complex(phase_real, phase_imag)
        signal = signal * c_phase
        
        return signal
    
    def reset(self):
        #nn.init_normal_(self.phase, 0, 0.02)
        torch.nn.init.constant_(self.phase, val = 0)

class optical_model(torch.nn.Module):
    def __init__(self, num_layers, plane_size, f_kernel, f_kernel_sub, aperture, num_classes):
        '''
        args:
            f_kernel: tensor, np.fft.fft2(np.fft.ifftshift(prop)), prop is the propagator generated from Raleigh Sommerfield equation
            sensor: sensor size
        '''
        super(optical_model, self).__init__()
        self.plane_size = plane_size
        self.num_layers = num_layers
        self.optical = optical_layer(plane_size)
        self.register_buffer('fk_const', f_kernel)
        self.register_buffer('fk_sub_const', f_kernel_sub)
        self.register_buffer('aperture_const', aperture)
        self.mask_const = torch.nn.Parameter(torch.empty((1, num_classes, plane_size, plane_size))).requires_grad_(requires_grad=False)

    def forward(self, signal, noise):
        
        #object free space prop to reach optical structure
        signal = fourier_conv(signal, self.fk_const)
        if not (self.fk_sub_const is None):
            signal = fourier_conv(signal, self.fk_sub_const)
        
        # phase modulation
        signal = self.optical(signal)
        if not (self.aperture_const is None):
            signal = signal * self.aperture_const
        
        #free space prop
        signal = fourier_conv(signal, self.fk_const)
        
        signal = signal.abs()**2

        if not (noise is None):
            prop_noise = noise[0]
            white_noise = noise[1]
            signal = signal * (1 + prop_noise) + white_noise
        
        detectors = signal * self.mask_const
        detectors = torch.mean(detectors, dim = (2, 3))
        
        return signal, detectors
    def reset(self):
        self.optical.reset()
        