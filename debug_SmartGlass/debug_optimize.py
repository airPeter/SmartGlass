import sys
module_path = 'C:/Users/94735/OneDrive - UW-Madison/My Projects/Navy_STTR/'
sys.path.insert(1, module_path)
import SmartGlass as SG
import numpy as np
import matplotlib.pyplot as plt
import torchvision

obj = SG.MNISTObj(font = 9, size = 80)
num_classes = 10
C = 50
R = 30
coos = []
for i in range(num_classes):
    theta = i * 2 * np.pi / num_classes
    x = C + R * np.cos(theta)
    y = C + R * np.sin(theta)
    coos.append([y, x])
coos = np.array(coos)
detector = SG.CirleDetector(radius= 5, coos = coos)
sim = SG.Coherent(
    wavelength = 1,
    num_layers= 1,
    size = 100,
    res = 2,
    prop_dis = 200,
    object = obj,
    num_classes= num_classes,
)
#init_phase = SG.lens_profile(sim.plane_size, sim.step_size, sim.prop_dis/2, sim.wavelength)
sim.init_model('cuda', init_phase = None, init_detector = detector)
batch_size = 64
data_path = 'dataset/MNIST/'
sim.optimze_optics(
    lr = 0.001, 
    beta = 0.001,
    batch_size = batch_size,
    epoches = 20,
    test_freq = 500,
    notes = 'April18_size100_baseline',
    mu_white_noise= 10, # >0
    data_path = data_path)


