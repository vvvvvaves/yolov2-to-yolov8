import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from utils import IoU
import numpy as np
from PIL import Image

class VOCDatasetV2(Dataset):
    def __init__(self, devkit_path, 
                 subsets = [('VOC2007', 'trainval'), ('VOC2012', 'trainval')], 
                 anchors = [], scales = [13], 
                 threshold_ignore_prediction = 0.5,
                 transforms = None,
                 dtype=None, device=None):
        super().__init__()
        self.devkit_path = devkit_path
        self.subsets = subsets
        self.anchors = anchors
        self.scales = scales
        self.threshold_ignore_prediction = threshold_ignore_prediction
        self.transforms = transforms
        self.dtype = dtype
        self.device = device

        self.object_placed = 0
        self.object_not_placed = 0

        self.all_labels = []
        for subset in self.subsets:
            subset_path = os.path.join(self.devkit_path, subset[0], 'ImageSets', 'Main', '{}.txt'.format(subset[1]))
            print(os.path.exists(subset_path), subset_path)
            with open(subset_path, 'r') as file:
                subset_labels = file.read().splitlines()
            self.all_labels.append(subset_labels)

        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']

    def __getitem__(self, idx):
        # ========= TEST ==========
        idx_copy = idx
        # ========= TEST ==========
        # get paths
        subset_idx = 0
        for subset_labels in self.all_labels:
            if idx < len(subset_labels):
                break
            else:
                subset_idx += 1
                idx -= len(subset_labels)

        if idx < 0 or subset_idx >= len(self.subsets):
            raise Exception("Index out of range.")

        # print(subset_idx, idx)
        image_path = os.path.join(self.devkit_path, self.subsets[subset_idx][0], 'JPEGImages', '{}.jpg'.format(self.all_labels[subset_idx][idx]))
        annotation_path = os.path.join(self.devkit_path, self.subsets[subset_idx][0], 'Annotations', '{}.xml'.format(self.all_labels[subset_idx][idx]))

        # print(os.path.exists(image_path), image_path)
        # print(os.path.exists(annotation_path), annotation_path)

        # get PIL image
        PIL_img = Image.open(image_path)

        # initialize tensors
        gt_out = [torch.zeros(len(self.anchors)*(5+len(self.classes)), scale, scale, dtype=self.dtype, device=self.device) for scale in self.scales]
        
        # parse annotations
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        bboxes = []
        for item in root.findall('./object'):
            bndbox = item.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            class_label = self.classes.index(item.find("name").text)

            bboxes.append([xmin, ymin, xmax, ymax, class_label])

        if self.transforms:
            np_img = np.array(PIL_img.convert("RGB"))
            transformed = self.transforms(image=np_img, bboxes=bboxes)
            image = transformed['image']
            image = image.type(self.dtype)
            image = image.to(self.device)
            img_d, img_h, img_w = image.shape
            bboxes = transformed['bboxes']
        else:
            return PIL_img, bboxes

        for box in bboxes:
            xmin, ymin, xmax, ymax, class_label = box
            class_label = int(class_label)

            # =========== TEST ==========
            # print('class_label ', class_label)
            # =========== TEST ==========
        
            obj_w = xmax - xmin
            obj_h = ymax - ymin

            obj_xc = xmax - obj_w / 2
            obj_yc = ymax - obj_h / 2

            for scale_idx, scale in enumerate(self.scales):
                cell_w = img_w / scale
                cell_h = img_h / scale

                cell_x = int(obj_xc / cell_w)
                cell_y = int(obj_yc / cell_h)

                # =========== TEST ==========
                # print('cell_x ', cell_x)
                # print('cell_y ', cell_y)
                # =========== TEST ==========
                
                
                obj_xc = (obj_xc % cell_w) / cell_w
                obj_yc = (obj_yc % cell_h) / cell_h

                obj_w = obj_w / cell_w
                obj_h = obj_h / cell_h

                bndbox = torch.tensor([obj_xc, obj_yc, obj_w, obj_h])
                
                # print((cell_x, cell_y), bndbox, self.classes[class_label])
                
                IoUs = torch.empty(len(self.anchors))
                for i, anchor in enumerate(self.anchors):
                    cell_aw = anchor[0] * scale
                    cell_ah = anchor[1] * scale
                    _anchor = torch.tensor([obj_xc, obj_yc, cell_aw, cell_ah])
                    IoUs[i] = IoU(bndbox, _anchor)

                anchors_argsort = torch.argsort(IoUs, descending=True)
                best_anchor = anchors_argsort[0]

                # =========== TEST ==========
                # print('best_anchor ', best_anchor)
                # =========== TEST ==========
                
                
                placement_0 = best_anchor*(5+len(self.classes))
                _objectness = (placement_0, cell_x, cell_y)

                taken = gt_out[scale_idx][_objectness] == 1                    
                if taken:
                    # =========== TEST ==========
                    self.object_not_placed += 1
                    # =========== TEST ==========
                    continue
                else:
                    gt_out[scale_idx][_objectness] = 1
                    gt_out[scale_idx][placement_0+1:placement_0+5, cell_x, cell_y] = bndbox
                    
                    label_placement = placement_0 + 1 + 4 + class_label
                    gt_out[scale_idx][label_placement] = 1

                    # =========== TEST ==========
                    self.object_placed += 1
                    # =========== TEST ==========

                # not the best anchors
                # =========== TEST ==========
                # print(anchors_argsort, IoUs)
                # =========== TEST ==========
                for anchor_idx in anchors_argsort[1:]:
                    if IoUs[anchor_idx] > self.threshold_ignore_prediction:
                        placement_0 = anchor_idx*(5+len(self.classes))
                        _objectness = (placement_0, cell_x, cell_y)
                        gt_out[scale_idx][_objectness] = -1
                        
                    
        return (image, gt_out) if len(self.scales) > 1 else (image, gt_out[0])
        
    def __len__(self):
        summed_len = 0
        for _subset in self.all_labels:
            summed_len += len(_subset)
        return summed_len