import os
import numpy as np
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import time

from utils import train, losses, metrics
from sklearn.model_selection import train_test_split
from dataset.dataset import MyDataset
from utils import metrics

import albumentations as albu
import matplotlib.pyplot as plt


images_dir = './data/CamVid/images'
masks_dir = './data/CamVid/masks'

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def normalize_input(x, **kwargs):
    return x / 255.0

    
def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title(), c='white')
        plt.imshow(image)
    plt.show()
    
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

# helper function for labeling multiclass masks
def labelVisualize(num_class, color_dict, img):
    img_out = img[0,:,:] if len(img.shape) == 3 else img
    img_out = np.zeros(img_out.shape + (3,))    # np.zeros(img.shape[-2:] + (3,))
    for i in range(num_class):
        img_out[img[i,:,:]==1, :] = COLOR_DICT[i]
    return img_out / 255



# load best saved checkpoint
best_model = torch.load('./models/repvggA2_g16_deploy.pth').cuda()

# create test dataset
x_test_dir = './data/CamVid/images/test'
y_test_dir = './data/CamVid/masks/test'

pipeline_prepro = get_preprocessing(normalize_input)

# class labels for cityscape dataset
CLASSES = ['car']
ALL_CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']


test_dataset = MyDataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(),
    preprocessing=pipeline_prepro,
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)


device = 'cuda'

# test dataset without transformations for image visualization
test_dataset_vis = MyDataset(
    x_test_dir, y_test_dir, 
    classes=CLASSES,
)

iou_metric = metrics.IoU(threshold=0.5)

for i in range(200):
    n = np.random.choice(len(test_dataset))
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, raw_gt_mask = test_dataset[n]
    print(image.shape)
    print(raw_gt_mask.shape)
    gt_mask = labelVisualize(len(CLASSES), COLOR_DICT, raw_gt_mask)
    
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
    start_time = time.time()
    pr_mask = best_model.predict(x_tensor)
    print("--- %s seconds ---" % (time.time() - start_time))
    iou_score = iou_metric(pr_mask, torch.from_numpy(raw_gt_mask).to(device))
    print(f"IOU = {iou_score.cpu().numpy()}")
    print(pr_mask.shape)
    pr_mask = labelVisualize(len(CLASSES), COLOR_DICT, pr_mask[0].cpu().numpy().round())    # round automacally means therehold = 0.5
        
    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask.squeeze(), 
        predicted_mask=pr_mask
    )