import torch
import torch.nn as nn
from torch.autograd.profiler import record_function
from torch.nn.modules.upsampling import Upsample

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
        with record_function("Bottleneck block"):
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
        with record_function("C2f block"):
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
        with record_function("ClassifyV2"):
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
        with record_function("ClassifyV8"):
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

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, 
                 device=None, dtype=None):
        super().__init__()
        self.hidden_channels = out_channels//2
        self.conv1 = Conv(in_channels, out_channels=self.hidden_channels, kernel_size=(1,1), stride=(1,1),
                          padding=(0,0), bias=False,
                          device=device, dtype=dtype)
        self.conv2 = Conv(self.hidden_channels * 4, out_channels=out_channels, kernel_size=(1,1), stride=(1,1), 
                          padding=(0,0), bias=False,
                          device=device, dtype=dtype)
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=(1,1), padding=kernel_size // 2)

    def forward(self, x):
        out = self.conv1(x)
        
        _l = [out]
        for i in range(3):
            out = self.max_pool(out)
            _l.append(out)
        out = torch.cat(_l, 1)
        
        out = self.conv2(out)
        return out

class FPN(nn.Module):
    variants = {'n': {'d': 0.34, 'w': 0.25, 'mc': 1024},
                's': {'d': 0.34, 'w': 0.50, 'mc': 1024},
                'm': {'d': 0.67, 'w': 0.75, 'mc': 768},
                'l': {'d': 1.00, 'w': 1.00, 'mc': 512},
                'x': {'d': 1.00, 'w': 1.25, 'mc': 512}}
    
    def __init__(self, residual_connection=False, 
                 CSP=True, add_hidden=True, variant='n',
                 device=None, dtype=None):
        super().__init__()

        if variant not in self.variants.keys():
            raise Exception("Invalid variant.")
            
        self.variant = variant
        self.mc = self.variants[self.variant]['mc']
        self.w = self.variants[self.variant]['w']
        self.d = self.variants[self.variant]['d']
        
        self.upsample = Upsample(scale_factor=2.0, mode='nearest')

        self.c2f_16 = C2f(self._ch(512)+self._ch(1024), out_channels=self._ch(512), n=self._d(3), residual_connection=residual_connection, 
                        CSP=CSP, add_hidden=add_hidden, bottleneck=1.0,
                        device=device, dtype=dtype)

        self.c2f_8 = C2f(self._ch(256)+self._ch(512), out_channels=self._ch(256), n=self._d(3), residual_connection=residual_connection, 
                        CSP=CSP, add_hidden=add_hidden, bottleneck=1.0,
                        device=device, dtype=dtype)
        
        
    def forward(self, out_8, out_16, out_32):
        out = self.upsample(out_32)

        out = torch.cat([out, out_16], 1)
        out_16 = self.c2f_16(out)

        out = self.upsample(out_16)
        out = torch.cat([out, out_8], 1)
        out = self.c2f_8(out)

        return out, out_16, out_32

    def _ch(self, ch):
        return int(min(ch, self.mc)*self.w)

    def _d(self, d):
        return int(d * self.d)

class PANet(nn.Module):
    variants = {'n': {'d': 0.34, 'w': 0.25, 'mc': 1024},
                's': {'d': 0.34, 'w': 0.50, 'mc': 1024},
                'm': {'d': 0.67, 'w': 0.75, 'mc': 768},
                'l': {'d': 1.00, 'w': 1.00, 'mc': 512},
                'x': {'d': 1.00, 'w': 1.25, 'mc': 512}}
    def __init__(self, residual_connection=False, 
                 CSP=True, add_hidden=True, variant='n',
                 device=None, dtype=None):
        super().__init__()

        if variant not in self.variants.keys():
            raise Exception("Invalid variant.")
            
        self.variant = variant
        self.mc = self.variants[self.variant]['mc']
        self.w = self.variants[self.variant]['w']
        self.d = self.variants[self.variant]['d']
        
        self.conv8_16 = Conv(self._ch(256), out_channels=self._ch(256), kernel_size=(3,3), stride=(2,2),
                          padding=(1,1), bias=False,
                          device=device, dtype=dtype)

        self.c2f_16 = C2f(self._ch(256)+self._ch(512), out_channels=self._ch(512), n=self._d(3), residual_connection=residual_connection, 
                        CSP=CSP, add_hidden=add_hidden, bottleneck=1.0,
                        device=device, dtype=dtype)

        self.conv16_32 = Conv(self._ch(512), out_channels=self._ch(512), kernel_size=(3,3), stride=(2,2),
                          padding=(1,1), bias=False,
                          device=device, dtype=dtype)

        self.c2f_32 = C2f(self._ch(1024)+self._ch(512), out_channels=self._ch(1024), n=self._d(3), residual_connection=residual_connection, 
                        CSP=CSP, add_hidden=add_hidden, bottleneck=1.0,
                        device=device, dtype=dtype)

    def _ch(self, ch):
        return int(min(ch, self.mc)*self.w)

    def _d(self, d):
        return int(d * self.d)

    def forward(self, out_8, out_16, out_32):
        out = self.conv8_16(out_8)

        out = torch.cat([out, out_16], 1)
        out_16 = self.c2f_16(out)

        out = self.conv16_32(out_16)
        out = torch.cat([out, out_32], 1)
        out = self.c2f_32(out)
        
        return out_8, out_16, out

class Detect(nn.Module):
    variants = {'n': {'d': 0.34, 'w': 0.25, 'mc': 1024},
                's': {'d': 0.34, 'w': 0.50, 'mc': 1024},
                'm': {'d': 0.67, 'w': 0.75, 'mc': 768},
                'l': {'d': 1.00, 'w': 1.00, 'mc': 512},
                'x': {'d': 1.00, 'w': 1.25, 'mc': 512}}
    def __init__(self, in_channels, decoupled=True,
                 num_classes=80, num_boxes=16, variant='n',
                 device=None, dtype=None):
        super().__init__()
        self.decoupled = decoupled
        
        if variant not in self.variants.keys():
            raise Exception("Invalid variant.")
        self.variant = variant
        self.mc = self.variants[self.variant]['mc']
        self.w = self.variants[self.variant]['w']
        self.d = self.variants[self.variant]['d']

        ch_0 = int(self.w*256)
        bbox_hidden, cls_hidden = max((16, ch_0 // 4, num_boxes * 4)), max(ch_0, min(num_classes, 100))

        if decoupled:
            self.bbox_conv1 = Conv(in_channels, out_channels=bbox_hidden, kernel_size=(3,3), stride=(1,1),
                              padding=(1,1), bias=False,
                              device=device, dtype=dtype)
    
            self.bbox_conv2 = Conv(bbox_hidden, out_channels=bbox_hidden, kernel_size=(3,3), stride=(1,1),
                              padding=(1,1), bias=False,
                              device=device, dtype=dtype)
            
            self.bbox_conv3 = nn.Conv2d(bbox_hidden, out_channels=num_boxes * 4, kernel_size=(1,1), stride=(1,1), 
                                        padding=(0,0), bias=True, 
                                        device=device, dtype=dtype)

            self.cls_conv1 = Conv(in_channels, out_channels=cls_hidden, kernel_size=(3,3), stride=(1,1),
                              padding=(1,1), bias=False,
                              device=device, dtype=dtype)
    
            self.cls_conv2 = Conv(cls_hidden, out_channels=cls_hidden, kernel_size=(3,3), stride=(1,1),
                              padding=(1,1), bias=False,
                              device=device, dtype=dtype)

            self.cls_conv3 = nn.Conv2d(cls_hidden, out_channels=num_classes, kernel_size=(1,1), stride=(1,1), 
                                        padding=(0,0), bias=True, 
                                        device=device, dtype=dtype)
        
        else:
            hidden = bbox_hidden + cls_hidden
            self.conv1 = Conv(in_channels, out_channels=hidden, kernel_size=(3,3), stride=(1,1),
                              padding=(1,1), bias=False,
                              device=device, dtype=dtype)

            self.conv2 = Conv(hidden, out_channels=hidden, kernel_size=(3,3), stride=(1,1),
                              padding=(1,1), bias=False,
                              device=device, dtype=dtype)

            self.conv3 = nn.Conv2d(hidden, out_channels=num_boxes * 4 + num_classes, kernel_size=(1,1), stride=(1,1), 
                                        padding=(0,0), bias=True, 
                                        device=device, dtype=dtype)

            
            

    def forward(self, x):
        if self.decoupled:
            out_bb = self.bbox_conv1(x)
            out_bb = self.bbox_conv2(out_bb)
            out_bb = self.bbox_conv3(out_bb)

            out_cls = self.cls_conv1(x)
            out_cls = self.cls_conv2(out_cls)
            out_cls = self.cls_conv3(out_cls)

            return {'bb': out_bb, 'cls': out_cls}
        else:
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)

            return out