import torch
import torch.nn as nn
from torch.autograd.profiler import record_function

class Conv(nn.Module): # for my experiments, it is identical to the ultralytics conv module
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride=1, padding=1, 
                 dilation=1, groups=1, bias=False, 
                 padding_mode='zeros', act='SiLU', device=None, dtype=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size, stride, padding, 
                             dilation, groups, bias, 
                             padding_mode, device, dtype)
        self.bn = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
        if act == 'Leaky':
            self.act = nn.LeakyReLU(0.1)
        elif act == 'SiLU':
            self.act = nn.SiLU()
        else:
            raise Exception("Invalid activation function.")

    def forward(self, x):
        with record_function("Conv block"):
            out = self.act(self.bn(self.conv(x)))
        return out

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 residual_connection=True, bottleneck=0.5,
                 device=None, dtype=None):
        super().__init__()
        self.hidden_channels = int(out_channels*bottleneck)
        self.conv1 = Conv(in_channels, out_channels=self.hidden_channels, kernel_size=(3,3), stride=(1,1),
                          padding=(1,1), bias=False,
                          device=device, dtype=dtype)
        self.conv2 = Conv(self.hidden_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), 
                          padding=(1,1), bias=False,
                          device=device, dtype=dtype)
        self.add = residual_connection and in_channels == out_channels

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add:
            return x + out
        else:
            return out

class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, residual_connection=False, CSP=False, add_hidden=False, bottleneck=1.0,
                device=None, dtype=None):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels=out_channels, kernel_size=(1,1), stride=(1,1), 
                          padding=(0,0), bias=False,
                          device=device, dtype=dtype)
        self.hidden_channels = out_channels // 2 if CSP else out_channels

        self.CSP = CSP
        self.add_hidden = CSP and add_hidden
        if self.add_hidden:
            self.conv2 = Conv((2 + n) * self.hidden_channels, out_channels=out_channels, kernel_size=(1,1), stride=(1,1), 
                              padding=(0,0), bias=False,
                              device=device, dtype=dtype)
        else:
            self.conv2 = Conv(out_channels, out_channels=out_channels, kernel_size=(1,1), stride=(1,1),
                              padding=(0,0), bias=False,
                              device=device, dtype=dtype)
        
        self.n_blocks = nn.ModuleList([
            Bottleneck(self.hidden_channels, self.hidden_channels, 
                       residual_connection=residual_connection, bottleneck=bottleneck, device=device, dtype=dtype) for _ in range(n)
        ])

    def forward(self, x):
        out = self.conv1(x)
        if self.CSP:
            _out = list(out.chunk(2, dim=1))
            out = _out[0]
            for block in self.n_blocks:
                out = block(out)
                if self.add_hidden:
                    _out.append(out)
            if not self.add_hidden:
                _out[0] = out
            out = torch.cat(_out, 1)
        else:
            for block in self.n_blocks:
                out = block(out)
        out = self.conv2(out)
        return out

class ClassifyV2(nn.Module):
    def __init__(self, in_channels, num_classes=10, device=None, dtype=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), 
                              padding=(0,0), bias=False,
                              device=device, dtype=dtype)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.conv(x)
        with record_function("Adaptive Average Pooling"):
            out = self.avg_pool(out)

        with record_function("Flatten"):
            out = out.flatten(1)
        
        with record_function("Log Softmax"):
            out = self.softmax(out)

        return out

class ClassifyV8(nn.Module):
    def __init__(self, in_channels, num_classes=10, device=None, dtype=None):
        super().__init__()
        self.hidden_channels = 1280
        self.conv = Conv(in_channels, out_channels=self.hidden_channels, kernel_size=(1, 1), stride=(1,1), 
                         padding=(0,0), bias=False,
                         device=device, dtype=dtype)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(self.hidden_channels, out_features=num_classes,
                               bias=True,
                               device=device, dtype=dtype)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if type(x) is list:
            x = torch.cat(x, 1)
        out = self.conv(x)
        with record_function("Adaptive Average Pooling"):
            out = self.pool(out)
            
        with record_function("Flatten"):
            out = out.flatten(1)
            
        with record_function("Linear layer"):
            out = self.linear(out)
            
        with record_function("Log Softmax"):
            out = self.softmax(out)
        return out 

