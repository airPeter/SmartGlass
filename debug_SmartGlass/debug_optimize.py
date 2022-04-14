import sys
module_path = 'C:/Users/94735/OneDrive - UW-Madison/My Projects/Navy_STTR/'
sys.path.insert(1, module_path)
import SmartGlass as SG
import numpy as np
import matplotlib.pyplot as plt
import torchvision

obj = SG.MNISTObj(font = 9, size = 80)
K = 10
C = 50
R = 30
coos = []
for i in range(K):
    theta = i * 2 * np.pi / K
    x = C + R * np.cos(theta)
    y = C + R * np.sin(theta)
    coos.append((y, x))
coos = np.array(coos)
coos = coos.T
detector = SG.CirleDetector(radius= 5, coos = coos)
sim = SG.Coherent(
    wavelength = 1,
    num_layers= 1,
    size = 100,
    res = 2,
    prop_dis = 200,
    object = obj,
    detector = detector
)
#init_phase = SG.lens_profile(sim.plane_size, sim.step_size, sim.prop_dis/2, sim.wavelength)
sim.init_model('cuda', None)
sim.optimze_optics(
    lr = 0.001, 
    beta = 0.001,
    batch_size = 32,
    epoches = 1,
    test_freq = 500,
    notes = 'April14_debug',
    mu_white_noise= 10, # >0
    data_path = 'dataset/MNIST/')


