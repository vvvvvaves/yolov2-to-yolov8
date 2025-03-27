import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pickle
with open('anchors_VOC0712trainval.pickle', 'rb') as handle:
    anchors = pickle.load(handle)

transforms = A.Compose([
    A.Resize(width=224, height=224),
    A.VerticalFlip(p=1.0),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc'))

device = torch.device('cuda:0')
dtype=torch.float32

# train_set = VOCDatasetV8(devkit_path = '../../datasets/VOCdevkit/', scales=[80, 40, 20], transforms=transforms, device=device, dtype=dtype)
train_set = VOCDatasetV2(devkit_path = '../../datasets/VOCdevkit/', scales=[20], anchors=anchors, transforms=transforms, device=device, dtype=dtype)

train_loader = DataLoader(train_set, batch_size=64, shuffle=False)

for img in train_loader:
    del img

print(train_set.object_placed, train_set.object_not_placed, train_set.object_placed + train_set.object_not_placed, train_set.total)





