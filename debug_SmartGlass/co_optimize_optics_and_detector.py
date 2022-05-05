import sys
module_path = 'C:/Users/94735/OneDrive - UW-Madison/My Projects/Navy_STTR/'
sys.path.insert(1, module_path)
import SmartGlass as SG
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os

def gen_circular_coos(C, R, num_classes):
    coos = []
    for i in range(num_classes):
        theta = i * 2 * np.pi / num_classes
        x = C + R * np.cos(theta)
        y = C + R * np.sin(theta)
        coos.append([y, x])
    coos = np.array(coos) 
    return coos

obj = SG.MNISTObj(font = 9, size = 80)
num_classes = 10
C = 50
R = 30
detector_r = 5
coos = gen_circular_coos(C, R, num_classes)
detector = SG.CirleDetector(radius= detector_r, coos = coos)
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
#phase = np.genfromtxt("output_coherent/April14_size100/phase.csv", delimiter = ',')
sim.init_model('cuda', init_phase = None, init_detector = detector)
batch_size = 64
data_path = 'dataset/MNIST/'
# test_samples = 10000
# select_idx = np.random.choice(np.arange(test_samples), size = (batch_size * 50,))
# for i in range(5):
#     out_path = 'output_coherent/April18_size100_co_optimze_beta0.0001_RanRestartHillClimb/'
#     if not os.path.exists(out_path):
#         os.mkdir(out_path)
#     sim.optimze_optics(
#         lr = 0.001, 
#         beta = 0.0001,
#         batch_size = batch_size,
#         epoches = 2,
#         test_freq = 500,
#         notes = 'opt_' + str(i),
#         mu_white_noise= 10, # >0
#         data_path = data_path,
#         out_path= out_path)
#     best_detector = sim.optimze_detector(batch_size, data_path, select_idx, n_iter = 200)
#     sim.assign_detector(best_detector)
init_coos = []
R_min = num_classes * 2 * detector_r / 2 / np.pi * 1.1
R_max = C - 2 * detector_r
print("R min, max", R_min, R_max)
population = 5
for i in range(population):
    tmp_R = R_min + i * (R_max - R_min) / population
    init_coos.append(np.round(gen_circular_coos(C, tmp_R, num_classes),1))
sim.optimze(
    lr = 0.001,
    beta = 0.001,
    batch_size= batch_size,
    epoches = 3,
    notes = 'April18_size100_train_each_iter_ESO',
    mu_white_noise= 10,
    data_path= data_path,
    init_coos = init_coos,
    subspace_size= 100,
    n_iter= 50,
    population = population
)