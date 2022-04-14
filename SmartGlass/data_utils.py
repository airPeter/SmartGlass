import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from torch.utils.data import Dataset
class SimpleDataset(Dataset):
    def __init__(self,X, Y, transform = None):
        self.X = X 
        self.Y = Y
        self.transform = transform
        
    def __getitem__(self,index): 
        sample = {'X':self.X[index], 'Y': self.Y[index]}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.X.shape[0]
    
def normal(x, y, ux, uy, sigma):
    output = np.exp(-((x-ux)**2 + (y - uy)**2)/(2*sigma**2))
    return output
def random_intensity_bias(img_size):
    sigma = 4
    x = np.linspace(-1,1,img_size)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    ux, uy = np.random.uniform(-2,2, size = (2,))
    normal_dis = normal(X, Y, ux, uy, sigma)
    return normal_dis

def map_label(labels, subset, label_map1):
    L = labels.shape[0]
    out_label = np.zeros((L,))
    out_group = out_label.copy()
    for i in range(L):
        label = labels[i]
        out_label[i] = label_map1[label]
        if label in subset:
            out_group[i] = 1
        else:
            out_group[i] = - 1
    return out_label, out_group
    
def input_label_process(x,y,plane_size, object_size, font, random_shift, background):
    
    img_resized = cv2.resize(x,(object_size, object_size))
    img_blured = cv2.cv2.GaussianBlur(img_resized,(font,font),0)
    img_blured[img_blured > 0.1] = 1
    img_blured[img_blured <= 0.1] = 0
    if background:
        img_blured = img_blured * random_intensity_bias(object_size)
    shift_x = (plane_size - object_size)//2
    if random_shift:
        shift = np.random.randint(-shift_x//2, shift_x//2, (2,)) + shift_x
    else:
        shift = np.array([shift_x, shift_x], dtype = int)
    output_tmp = np.zeros((plane_size, plane_size))
    output_tmp[shift[0]: shift[0] + object_size, shift[1]: shift[1] + object_size] = img_blured
    output_x = np.reshape(output_tmp, (1, plane_size, plane_size))
    return output_x, y

def data_generator(batch_size, data, data_label, group, shuffle):
    while(True):
        indexes = np.arange(0,data.shape[0],1)
        if shuffle == True:
            np.random.shuffle(indexes)
        max_range = int(data.shape[0]/batch_size)
        for i in range(max_range):

            data_temp = np.array([data[k] for k in indexes[i*batch_size:(i+1)*batch_size]])
            data_label_temp = np.array([data_label[k] for k in indexes[i*batch_size:(i+1)*batch_size]])
            group_temp = np.array([group[k] for k in indexes[i*batch_size:(i+1)*batch_size]])
            yield data_temp, data_label_temp, group_temp

# def detector_pos(img):
#     gap = 175
#     gap_m = 120
#     sensor = 100

#     pos = {
#     0: (gap, gap), 1: (gap, gap*2 + sensor), 2: (gap, gap * 3 + sensor * 2),
#     3: (gap*2 + sensor, gap_m), 4: (gap*2 + sensor, gap_m*2 + sensor),
#     5: (gap*2 + sensor, gap_m * 3 + sensor * 2), 6: (gap*2 + sensor, gap_m * 4 + sensor * 3),
#     7: (gap*3 + sensor*2, gap), 8: (gap*3 + sensor*2, gap*2 + sensor), 
#     9: (gap*3 + sensor*2, gap*3 + sensor * 2)
#     }
#     thickness = 5
#     color = (192,192, 192)
#     for i in range(10):
#         img = cv2.rectangle(img, (pos[i][1], pos[i][0]), (pos[i][1] + sensor, pos[i][0] + sensor), color, thickness)
#     return img

def plt_detector(ax):
    # gap = 175
    # gap_m = 120
    # sensor = 100
    gap = 100
    gap_m = 40
    sensor = 200
    fig_size = 1000
    pos = {
    0: (gap, gap), 1: (gap, gap*2 + sensor), 2: (gap, gap * 3 + sensor * 2),
    3: (gap*2 + sensor, gap_m), 4: (gap*2 + sensor, gap_m*2 + sensor),
    5: (gap*2 + sensor, gap_m * 3 + sensor * 2), 6: (gap*2 + sensor, gap_m * 4 + sensor * 3),
    7: (gap*3 + sensor*2, gap), 8: (gap*3 + sensor*2, gap*2 + sensor), 
    9: (gap*3 + sensor*2, gap*3 + sensor * 2)
    }
    thickness = 2
    for i in range(10):
        ax.add_patch(plt.Rectangle((pos[i][1], pos[i][0]), sensor, sensor, lw = thickness, ec = "gray", fc = "None"))
    return    

def create_circular_detector(r, Ks, plane_size):
    num_detectors = Ks.shape[1]
    out = np.zeros((1, num_detectors, plane_size, plane_size))
    for i in range(num_detectors):
        x = Ks[1,i]
        y = Ks[0,i]
        out[0,i] = cv2.circle(out[0,i], (int(x), int(y)), int(r), 1, -1)
    return out

def draw_circular_detector(img, r, Ks):
    color = img.max()//2
    num_detectors = Ks.shape[1]
    for i in range(num_detectors):
        x = Ks[1,i]
        y = Ks[0,i]
        img = cv2.circle(img, (int(x), int(y)), int(r), color, 3)
    return img

def save_image(img, path, title, colorbar):
    fig, ax = plt.subplots()
    plot = ax.imshow(img)
    if colorbar:
        plt.colorbar(plot, ax = ax)
        #plt_detector(ax)

    plt.title(title)
    plt.savefig(path)
    plt.close(fig)
    return

def save_npy(img, path):
    np.save(path, img)
    return

def save_batch(name, img_batch, pred_label, label,group_label, gl_step, image_path, detector_paras, npy_path = False):
    
    for i in range(img_batch.shape[0]):
        toi_path = image_path + gl_step + 'index_' + str(i) + name +'.png'
        toi_title = 'predict_label: '+ str(pred_label[i]) + '_true_label: ' + str(label[i]) + '_group_label: ' + str(group_label[i])

        if name == 'tra_image':
            img = draw_circular_detector(img_batch[i], **detector_paras)
            save_image(img,toi_path, toi_title, True)
        else:
            save_image(img_batch[i],toi_path, toi_title, False) 

        if npy_path:
            path = npy_path + gl_step + 'index_' + str(i) + name +'.npy'
            save_npy(img_batch[i], path)

    return

def save_to_excel(data, name, plane_size, path):

    data = np.reshape(data, (plane_size, plane_size))
    data_df = pd.DataFrame(data)
    data_df.to_excel(path + 'excel_data/'+ name + '.xlsx', index = False, header = False)


def CM(y_test, y_test_pred, classes, path):

    cm = confusion_matrix(y_test, y_test_pred, labels = np.arange(classes))
    # Calculate and show correlation matrix
    sns.set(font_scale=1)
    hm = sns.heatmap(cm,
                    cbar=True,
                    annot=True,
                    square=True,
                    fmt='d',
                    annot_kws={'size': 8},
                    yticklabels=np.arange(classes),
                    xticklabels=np.arange(classes))
    plt.xlabel('prediction')
    plt.ylabel('ground_truth')
    plt.savefig(path+'/' + 'confusion_matrix.png')

def vis_img(img, title):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.title(title)
    plt.show()