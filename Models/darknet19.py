import torch
import torch.nn as nn
from collections import OrderedDict
from Models.block import Conv, ClassifyV2
from torch.autograd.profiler import record_function

class Darknet19(torch.nn.Module):
    def __init__(self, device=None, dtype=None, num_classes=10, act='Leaky'):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = Conv(3, out_channels=32, kernel_size=(3,3), stride=(1,1), 
                          padding=(1,1), bias=False, act=act,
                          device=device, dtype=dtype)
        self.conv2 = Conv(32, out_channels=64, kernel_size=(3,3), stride=(1,1), 
                          padding=(1,1), bias=False, act=act,
                          device=device, dtype=dtype)

        self.seq3_5 = nn.Sequential(OrderedDict([
            ('conv3', Conv(64, out_channels=128, kernel_size=(3,3), stride=(1,1), 
                          padding=(1,1), bias=False, act=act,
                          device=device, dtype=dtype)),
            ('conv4', Conv(128, out_channels=64, kernel_size=(1,1), stride=(1,1), 
                          padding=(0,0), bias=False, act=act,
                          device=device, dtype=dtype)),
            ('conv5', Conv(64, out_channels=128, kernel_size=(3,3), stride=(1,1), 
                          padding=(1,1), bias=False, act=act,
                          device=device, dtype=dtype))
        ]))
        
        self.seq6_8 = nn.Sequential(OrderedDict([
            ('conv6', Conv(128, out_channels=256, kernel_size=(3,3), stride=(1,1), 
                          padding=(1,1), bias=False, act=act,
                          device=device, dtype=dtype)),
            ('conv7', Conv(256, out_channels=128, kernel_size=(1,1), stride=(1,1), 
                          padding=(0,0), bias=False, act=act,
                          device=device, dtype=dtype)),
            ('conv8', Conv(128, out_channels=256, kernel_size=(3,3), stride=(1,1), 
                          padding=(1,1), bias=False, act=act,
                          device=device, dtype=dtype))
        ]))

        self.seq9_13 = nn.Sequential(OrderedDict([
            ('conv9', Conv(256, out_channels=512, kernel_size=(3,3), stride=(1,1), 
                          padding=(1,1), bias=False, act=act,
                          device=device, dtype=dtype)),
            ('conv10', Conv(512, out_channels=256, kernel_size=(1,1), stride=(1,1), 
                          padding=(0,0), bias=False, act=act,
                          device=device, dtype=dtype)),
            ('conv11', Conv(256, out_channels=512, kernel_size=(3,3), stride=(1,1), 
                          padding=(1,1), bias=False, act=act,
                          device=device, dtype=dtype)),
            ('conv12', Conv(512, out_channels=256, kernel_size=(1,1), stride=(1,1), 
                          padding=(0,0), bias=False, act=act,
                          device=device, dtype=dtype)),
            ('conv13', Conv(256, out_channels=512, kernel_size=(3,3), stride=(1,1), 
                          padding=(1,1), bias=False, act=act,
                          device=device, dtype=dtype))
        ]))

        self.seq14_18 = nn.Sequential(OrderedDict([
            ('conv14', Conv(512, out_channels=1024, kernel_size=(3,3), stride=(1,1), 
                          padding=(1,1), bias=False, act=act,
                          device=device, dtype=dtype)),
            ('conv15', Conv(1024, out_channels=512, kernel_size=(1,1), stride=(1,1), 
                          padding=(0,0), bias=False, act=act,
                          device=device, dtype=dtype)),
            ('conv16', Conv(512, out_channels=1024, kernel_size=(3,3), stride=(1,1), 
                          padding=(1,1), bias=False, act=act,
                          device=device, dtype=dtype)),
            ('conv17', Conv(1024, out_channels=512, kernel_size=(1,1), stride=(1,1), 
                          padding=(0,0), bias=False, act=act,
                          device=device, dtype=dtype)),
            ('conv18', Conv(512, out_channels=1024, kernel_size=(3,3), stride=(1,1), 
                          padding=(1,1), bias=False, act=act,
                          device=device, dtype=dtype))
        ]))

        self.classify = ClassifyV2(in_channels=1024, num_classes=num_classes, 
                                   device=device, dtype=dtype)


    def forward(self, x):
        # 224x224, stride: 0
        out = self.conv1(x)
        with record_function("Max pooling"):
            out = self.max_pool(out)

        # 112x112, stride: 2
        out = self.conv2(out)
        with record_function("Max pooling"):
            out = self.max_pool(out)

        # 56x56, stride: 4
        out = self.seq3_5(out)
        with record_function("Max pooling"):
            out = self.max_pool(out)

        # 28x28, stride: 8
        out = self.seq6_8(out)
        with record_function("Max pooling"):
            out = self.max_pool(out)

        # 14x14, stride: 16
        out = self.seq9_13(out)
        with record_function("Max pooling"):
            out = self.max_pool(out)

        # 7x7, stride: 32
        out = self.seq14_18(out)

        # classification head
        out = self.classify(out)

        return out