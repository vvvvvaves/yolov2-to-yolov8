import torch
import torch.nn as nn
from Models.block import Conv, C2f, SPPF, FPN, PANet, Detect
from torch.autograd.profiler import record_function

class Model(nn.Module):
    variants = {'n': {'d': 0.34, 'w': 0.25, 'mc': 1024},
                's': {'d': 0.34, 'w': 0.50, 'mc': 1024},
                'm': {'d': 0.67, 'w': 0.75, 'mc': 768},
                'l': {'d': 1.00, 'w': 1.00, 'mc': 512},
                'x': {'d': 1.00, 'w': 1.25, 'mc': 512}}

    def __init__(self, three_heads=True, decoupled=True,
                 _FPN=True, _PANet=True, _SPPF=True,
                 num_classes=80, num_boxes=16, variant='n', 
                 device=None, dtype=None):
        super().__init__()
        self.three_heads = three_heads
        self._FPN = _FPN
        self._PANet = _PANet
        self._SPPF = _SPPF

        # Backbone model parameters
        residual_connection = True
        CSP = True
        add_hidden = True
        bottleneck = 1.0
        
        if variant not in self.variants.keys():
            raise Exception("Invalid variant.")
        self.variant = variant
        self.mc = Model.variants[self.variant]['mc']
        self.w = Model.variants[self.variant]['w']
        self.d = Model.variants[self.variant]['d']
        

        self.conv1 = Conv(3, out_channels=self._ch(64), kernel_size=(3, 3), stride=(2, 2), 
                         padding=(1, 1), bias=False, 
                         device=device, dtype=dtype)
        self.conv2 = Conv(self._ch(64), out_channels=self._ch(128), kernel_size=(3, 3), stride=(2, 2), 
                          padding=(1, 1), bias=False,
                          device=device, dtype=dtype)
        self.c2f1 = C2f(self._ch(128), out_channels=self._ch(128), n=self._d(3), residual_connection=residual_connection, 
                        CSP=CSP, add_hidden=add_hidden, bottleneck=1.0,
                        device=device, dtype=dtype)
        self.conv3 = Conv(self._ch(128), out_channels=self._ch(256), kernel_size=(3, 3), stride=(2, 2), 
                         padding=(1, 1), bias=False, 
                         device=device, dtype=dtype)
        self.c2f2 = C2f(self._ch(256), out_channels=self._ch(256), n=self._d(6), residual_connection=residual_connection, 
                        CSP=CSP, add_hidden=add_hidden, bottleneck=1.0,
                        device=device, dtype=dtype)
        # c2f2 --- stride 8 ---->
        
        self.conv4 = Conv(self._ch(256), out_channels=self._ch(512), kernel_size=(3, 3), stride=(2, 2), 
                         padding=(1, 1), bias=False, 
                         device=device, dtype=dtype)
        self.c2f3 = C2f(self._ch(512), out_channels=self._ch(512), n=self._d(6), residual_connection=residual_connection, 
                        CSP=CSP, add_hidden=add_hidden, bottleneck=1.0,
                        device=device, dtype=dtype)
        # c2f3 --- stride 16 ---->
        
        self.conv5 = Conv(self._ch(512), out_channels=self._ch(1024), kernel_size=(3, 3), stride=(2, 2), 
                         padding=(1, 1), bias=False, 
                         device=device, dtype=dtype)
        self.c2f4 = C2f(self._ch(1024), out_channels=self._ch(1024), n=self._d(3), residual_connection=residual_connection, 
                        CSP=CSP, add_hidden=add_hidden, bottleneck=1.0,
                        device=device, dtype=dtype)

        if self._SPPF:
            self.sppf = SPPF(self._ch(1024), out_channels=self._ch(1024), kernel_size=5,
                            device=device, dtype=dtype)
        # sppf --- stride 32 ---->

        if self._FPN:
            self.fpn = FPN(residual_connection=False, 
                           CSP=CSP, add_hidden=add_hidden, 
                           variant=self.variant,
                           device=device, dtype=dtype)

        if self._PANet:
            self.panet = PANet(residual_connection=False, 
                           CSP=CSP, add_hidden=add_hidden, 
                           variant=self.variant,
                           device=device, dtype=dtype)

        if self.three_heads:
            self.detect_8 = Detect(self._ch(256), decoupled=decoupled,
                                   num_classes=num_classes, num_boxes=num_boxes, 
                                   variant=self.variant, device=device, dtype=dtype)
    
            self.detect_16 = Detect(self._ch(512), decoupled=decoupled,
                                   num_classes=num_classes, num_boxes=num_boxes, 
                                   variant=self.variant, device=device, dtype=dtype)
    
        self.detect_32 = Detect(self._ch(1024), decoupled=decoupled,
                               num_classes=num_classes, num_boxes=num_boxes, 
                               variant=self.variant, device=device, dtype=dtype)

    
    def _ch(self, ch):
        return int(min(ch, self.mc)*self.w)

    def _d(self, d):
        return int(d * self.d)

    def forward(self, x):
        with record_function('conv1'):
            out = self.conv1(x)
        
        with record_function('conv2'):
            out = self.conv2(out)
        with record_function('c2f1'):
            out = self.c2f1(out)

        with record_function('conv3'):
            out = self.conv3(out)
        with record_function('c2f2'):
            out = self.c2f2(out)
        
        # c2f2 --- stride 8 ---->
        out_8 = out
            
        with record_function('conv4'):
            out = self.conv4(out)
        with record_function('c2f3'):
            out = self.c2f3(out)

        # c2f3 --- stride 16 ---->
        out_16 = out
            
        with record_function('conv5'):
            out = self.conv5(out)
        with record_function('c2f4'):
            out = self.c2f4(out)

        if self._SPPF:
            with record_function('sppf'):
                out = self.sppf(out)

        # sppf --- stride 32 ---->
        if self._FPN:
            with record_function('fpn'):
                out_8, out_16, out = self.fpn(out_8, out_16, out)

        
        if self._PANet:
            with record_function('panet'):
                out_8, out_16, out = self.panet(out_8, out_16, out)

        arranged = None
        if self.three_heads:
            with record_function('detect'):
                out_8 = self.detect_8(out_8)
                out_16 = self.detect_16(out_16)
                out = self.detect_32(out)

            arranged = {'out_8': out_8, 'out_16': out_16, 'out_32': out}
            return arranged
        else:
            with record_function('detect'):
                out = self.detect_32(out)

            return out