import numpy as np
import os
import h5py

import cv2

import matplotlib.pyplot as plt
import json

# sys.path.append('..') #Hack add ROOT DIR
# from baseconfig import CONF

# dataset_dir = CONF.PATH.DETECTTRAIN
# dataset_dir = CONF.PATH.DETECTVAL

dataset_dir = os.path.join(os.getcwd(), "data/sample")
scenes = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

cls_to_objs = {
        0: 'chair',
        1: 'table',
        2: 'sofa',
        3: 'bed',
        4: 'tv_stand',
        5: 'cooler',
        6: 'night_stand'
    }

num_objs = [0, 0, 0, 0, 0, 0, 0]

for scene in scenes:
    json_file = os.path.join(dataset_dir, scene, "coco_data/coco_annotations.json")

    with open(json_file) as f:
        imgs_anns = json.load(f)

    for idx, v in enumerate(imgs_anns['annotations']):
        if v['image_id'] == 0:
            num_objs[v['category_id']-1] = num_objs[v['category_id']-1] + 1

print("Number of scenes examined:", len(scenes))

count = 0
for idx, num in enumerate(num_objs):
    count = count + num
    print("Number of {}: {}".format(cls_to_objs[idx], num))

print("Average number of objects per scene:", count / len(scenes))
