import os
import cv2
import numpy as np
import torch

ALL_CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

class MyDataset(torch.utils.data.Dataset):
    
    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
            aux_loss=None
    ):
        self.all_img_names = os.listdir(images_dir)
        self.all_img_path = [os.path.join(
            images_dir, img_name) for img_name in self.all_img_names]
        self.all_mask_path = [os.path.join(
            masks_dir, img_name) for img_name in self.all_img_names]
        # convert str names to class values based on masks data
        self.class_values = [ALL_CLASSES.index(cls.lower()) for cls in classes]
        # self.class_values = [i for i in range(len(classes_name))]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.all_img_path[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = np.float32(image)
        mask = cv2.imread(self.all_mask_path[i], 0)
        if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
            raise (RuntimeError("Image & mask shape mismatch: " +
                                self.all_img_path[i] + " " + self.all_mask_path[i] + "\n"))
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        # masks.append(np.logical_not(masks[0]))
        mask = np.stack(masks, axis=-1).astype('float')
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
                raise (RuntimeError("Image & mask shape mismatch after annotation: " +
                                    self.all_img_path[i] + " " + self.all_mask_path[i] + "\n"))
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # return labels for auxiliary loss
        # if aux_loss:
        #    return image, mask, label
        return image, mask

    def __len__(self):
        return len(self.all_img_names)
