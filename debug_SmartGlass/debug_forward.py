import sys
module_path = 'C:/Users/94735/OneDrive - UW-Madison/My Projects/Navy_STTR/'
sys.path.insert(1, module_path)
import SmartGlass as SG
import numpy as np
import matplotlib.pyplot as plt
import torchvision

obj = SG.MNISTObj(font = 9, size = 40)
num_classes = 10
C = 25
R = 15
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
    size = 50,
    res = 2,
    prop_dis = 100,
    object = obj,
    num_classes= num_classes,
)
init_phase = SG.lens_profile(sim.plane_size, sim.step_size, sim.prop_dis/2, sim.wavelength)
sim.init_model('cuda', init_phase, detector)

mnist = torchvision.datasets.MNIST('dataset/', train= False, download = True)
x, y = mnist[0]
x = np.array(x)/255
SG.vis_img(img = x, title = str(y))
obj = sim.gen_obj(x)
SG.vis_img(np.squeeze(obj), "transform image to object.")
img = sim.forward(obj)
SG.vis_img(np.squeeze(img), "image of the object.")
