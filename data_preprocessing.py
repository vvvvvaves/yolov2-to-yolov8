import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import transforms
from PIL import Image

import json
import os

class ImageNetSubset(Dataset):
    def __init__(self, path='../datasets/', train=True, transform=None, half=False, show=False) -> None:
        self.show = show
        self.train = train
        self.dataset_path = path
        self.image_paths = self.get_dataset_paths(path)
        remove_labels_with_no_data(dataset_path=path)
        self.labels_str = self.get_labels_str()
        self.labels_train = self.get_labels_train()
        self.transform = transform
        self.half = half

    def get_labels_str(self):
        with open(os.path.join(self.dataset_path,'Labels_subset.json'), 'r') as file:
            labels_str = json.load(file)
        return labels_str

    def get_labels_train(self):
        if self.labels_str is None:
            self.get_labels_str()
        labels_train = {folder: idx for idx, folder in enumerate(list(self.labels_str.keys()))}
        return labels_train
            
    def get_dataset_paths(self, path):
        image_paths = []
        for root, dirs, files in os.walk(os.path.join(self.dataset_path, 'Train.X' if self.train else 'Val.X')):
            for file in files:
                image_paths.append(os.path.join(root, file).replace("\\","/"))
        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert('RGB') # cause some pictures had a shape of [1, q, h]
        folder_name = image_path.split('/')[-2]
        if self.show:
            label = self.labels_str[folder_name]
        else:
            label = self.labels_train[folder_name]
            label = torch.tensor(label)

        if self.transform:
            img = self.transform(img)
            if self.half:
                img = img.half()
                if not self.show:
                     label = label.half()
            return img, label
        else:
            if not self.show:
                to_tensor = transforms.ToTensor()
                img = to_tensor(img)
                if self.half:
                    img = img.half()
                    label = label.half()
            return img, label

def show(img):
    npimg = img.to(torch.float32).numpy() 
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def get_means(path, train_loader):
    means = None
    if os.path.isfile(path):
        with open(path, 'r') as file:
            means = json.load(file)['means']
    else:
        n = 0
        _sum = 0
        total_images = 0
        for i, (batch, labels) in enumerate(train_loader):
            imgs_t = batch.view(3, -1)
            _sum += imgs_t.sum(dim=1)
            n += imgs_t.shape[1]
            total_images += len(batch)
            print(f'batch {i+1} done. Total images: {total_images}')
        means = _sum / n
    print(f"Means are: {means}")
    return means if type(means) == list else means.tolist()

def get_stds(path, means, train_loader):
    stds = None
    if os.path.isfile(path):
        with open(path, 'r') as file:
            stds = json.load(file)['stds']
    else:
        n = 0
        _sum = 0
        total_images = 0
        for i, (batch, labels) in enumerate(train_loader):
            imgs_t = batch.view(3, -1)
            _sub = torch.sub(imgs_t, means.unsqueeze(1))
            _pow = torch.pow(_sub, 2)
            _sum += (_pow).sum(dim=1)
            n += imgs_t.shape[1]
            total_images += len(batch)
            print(f'batch {i+1} done. Total images: {total_images}')
        stds = torch.sqrt(_sum / (n-1))
    print(f"stds are: {stds}")
    return stds if type(stds) == list else stds.tolist()

def save_norms(means, stds, path='../datasets/norms.json'):
    norms = {}
    norms['means'] = means
    norms['stds'] = stds
    with open(path, 'w') as fp:
        json.dump(norms, fp)

def get_norms(path='../datasets/norms.json', train_loader=None):
    norms = None
    if os.path.isfile(path):
        with open(path, 'r') as file:
            norms = json.load(file)
    else:
        means = get_means(path, train_loader)
        stds = get_stds(path, train_loader)
        save_norms(means, stds, path)
        norms = {}
        norms['means'] = means
        norms['stds'] = stds
    return norms

def remove_labels_with_no_data(dataset_path='../datasets/'):
    _dirs = None
    for root, dirs, files in os.walk(os.path.join(dataset_path, 'Train.X')):
        _dirs = dirs
        break

    with open(os.path.join(dataset_path,'Labels.json'), 'r') as file:
        labels = json.load(file)

    labels_subset = {key: labels[key] for key in _dirs}
    with open(os.path.join(dataset_path, 'Labels_subset.json'), 'w') as fp:
        json.dump(labels_subset, fp)