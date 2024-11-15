import torch
import torch.nn as nn
from Models.block import *

class Model(nn.Module):
    variants = {'n': {'d': 0.34, 'w': 0.25, 'mc': 1024},
                's': {'d': 0.34, 'w': 0.50, 'mc': 1024},
                'm': {'d': 0.67, 'w': 0.75, 'mc': 768},
                'l': {'d': 1.00, 'w': 1.00, 'mc': 512},
                'xl': {'d': 1.00, 'w': 1.25, 'mc': 512}}

    def __init__(self, device=None, dtype=None, 
                 residual_connection=False, CSP=False, add_hidden=False, bottleneck=1.0,
                 num_classes=1000, variant='n', classifyV8=False):
        super().__init__()
        
        if variant not in Model.variants.keys():
            raise Exception("Invalid variant.")
        self.variant = variant
        self.mc = Model.variants[self.variant]['mc']
        self.w = Model.variants[self.variant]['w']
        self.d = Model.variants[self.variant]['d']
        self._ch = lambda ch: int(min(ch, self.mc)*self.w)
        self._d = lambda d: int(d * self.d)

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
        self.conv4 = Conv(self._ch(256), out_channels=self._ch(512), kernel_size=(3, 3), stride=(2, 2), 
                         padding=(1, 1), bias=False, 
                         device=device, dtype=dtype)
        self.c2f3 = C2f(self._ch(512), out_channels=self._ch(512), n=self._d(6), residual_connection=residual_connection, 
                        CSP=CSP, add_hidden=add_hidden, bottleneck=1.0,
                        device=device, dtype=dtype)
        self.conv5 = Conv(self._ch(512), out_channels=self._ch(1024), kernel_size=(3, 3), stride=(2, 2), 
                         padding=(1, 1), bias=False, 
                         device=device, dtype=dtype)
        self.c2f4 = C2f(self._ch(1024), out_channels=self._ch(1024), n=self._d(3), residual_connection=residual_connection, 
                        CSP=CSP, add_hidden=add_hidden, bottleneck=1.0,
                        device=device, dtype=dtype)

        if classifyV8:
            self.classify = ClassifyV8(self._ch(1024), num_classes=num_classes,
                                      device=device, dtype=dtype)
        else:
            self.classify = ClassifyV2(self._ch(1024), num_classes=num_classes,
                                      device=device, dtype=dtype)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        out = self.c2f1(out)
        out = self.conv3(out)

        out = self.c2f2(out)
        out = self.conv4(out)

        out = self.c2f3(out)
        out = self.conv5(out)

        out = self.c2f4(out)

        out = self.classify(out)
        return out