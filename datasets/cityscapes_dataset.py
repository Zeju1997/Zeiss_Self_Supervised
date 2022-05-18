# Cityscapes dataset (target)

import os
import os.path as osp
import sys

import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle']

    
class CityscapesDataset(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(256, 256), mean=(128, 128, 128), ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.id2label = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
            3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
            7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
            14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
            18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

        #self.id_to_trainid = {7: 1, 24: 2, 26: 3} #Road/car/people
        self.id_to_trainid = {11: 1, 24: 2, 21: 3} #Building/car/vegetation
        #self.ignore_label = 0

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        image_root = osp.join(self.root, 'cityscapes')
        image = Image.open(osp.join(image_root, "%s" % name)).convert('RGB')
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)

        image = np.asarray(image, np.float32)
        size = image.shape
        #image = image[:, :, ::-1]  # change to BGR
        #image -= self.mean
        image = image.transpose((2, 0, 1))
        sample = {'image': image.copy()}

        return sample
    
