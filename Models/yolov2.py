import torch
import torch.nn as nn

"""
Adapted from https://github.com/yjh0410/yolov2-yolov3_PyTorch .
The weights for the backbone are taken from https://github.com/yjh0410/pytorch-imagenet
"""

class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1, 
                 dtype=None, device=None):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.dtype = dtype
        self.device = device
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation, device=self.device, dtype=self.dtype),
            nn.BatchNorm2d(out_channels, device=self.device, dtype=self.dtype),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)

class reorg_layer(nn.Module):
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride
        
        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)

        return x

class DarkNet_19(nn.Module):
    def __init__(self, dtype=None, device=None):        
        super(DarkNet_19, self).__init__()
        self.dtype = dtype
        self.device = device
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1, 
                              device=self.device, dtype=self.dtype),
            nn.MaxPool2d((2,2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1, 
                              device=self.device, dtype=self.dtype),
            nn.MaxPool2d((2,2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1, 
                              device=self.device, dtype=self.dtype),
            Conv_BN_LeakyReLU(128, 64, 1, 
                              device=self.device, dtype=self.dtype),
            Conv_BN_LeakyReLU(64, 128, 3, 1, 
                              device=self.device, dtype=self.dtype),
            nn.MaxPool2d((2,2), 2)
        )

        # output : stride = 8, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1, 
                              device=self.device, dtype=self.dtype),
            Conv_BN_LeakyReLU(256, 128, 1, 
                              device=self.device, dtype=self.dtype),
            Conv_BN_LeakyReLU(128, 256, 3, 1, 
                              device=self.device, dtype=self.dtype),
        )


        # output : stride = 16, c = 512
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1, 
                              device=self.device, dtype=self.dtype),
            Conv_BN_LeakyReLU(512, 256, 1, 
                              device=self.device, dtype=self.dtype),
            Conv_BN_LeakyReLU(256, 512, 3, 1, 
                             device=self.device, dtype=self.dtype),
            Conv_BN_LeakyReLU(512, 256, 1,
                             device=self.device, dtype=self.dtype),
            Conv_BN_LeakyReLU(256, 512, 3, 1,
                             device=self.device, dtype=self.dtype),
        )
        
        # output : stride = 32, c = 1024
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1, 
                              device=self.device, dtype=self.dtype),
            Conv_BN_LeakyReLU(1024, 512, 1,
                             device=self.device, dtype=self.dtype),
            Conv_BN_LeakyReLU(512, 1024, 3, 1,
                             device=self.device, dtype=self.dtype),
            Conv_BN_LeakyReLU(1024, 512, 1,
                             device=self.device, dtype=self.dtype),
            Conv_BN_LeakyReLU(512, 1024, 3, 1,
                             device=self.device, dtype=self.dtype)
        )


    def forward(self, x):
        c1 = self.conv_1(x)
        c2 = self.conv_2(c1)
        c3 = self.conv_3(c2)
        c3 = self.conv_4(c3)
        c4 = self.conv_5(self.maxpool_4(c3))
        c5 = self.conv_6(self.maxpool_5(c4))

        output = {
            'layer1': c3,
            'layer2': c4,
            'layer3': c5
        }

        return output

class YOLOv2D19(nn.Module):
    def __init__(self, num_classes=20, num_anchors=5, state_dict_path='./darknet19_72.96.pth', device=None, dtype=None):
        super(YOLOv2D19, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.device = device
        self.dtype = dtype

        # Load pretrained backbone
        state_dict = torch.load(state_dict_path, map_location=self.device)
        del state_dict['conv_7.weight']
        del state_dict['conv_7.bias']

        self.backbone = DarkNet_19()
        self.backbone.load_state_dict(state_dict)
        
        # detection head
        self.convsets_1 = nn.Sequential(
            Conv_BN_LeakyReLU(1024, 1024, 3, 1,
                             device=self.device, dtype=self.dtype),
            Conv_BN_LeakyReLU(1024, 1024, 3, 1,
                             device=self.device, dtype=self.dtype)
        )

        self.route_layer = Conv_BN_LeakyReLU(512, 64, 1,
                                            device=self.device, dtype=self.dtype)
        self.reorg = reorg_layer(stride=2)

        self.convsets_2 = Conv_BN_LeakyReLU(1280, 1024, 3, 1,
                                           device=self.device, dtype=self.dtype)
        
        # prediction layer
        self.pred = nn.Conv2d(1024, self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1, 
                              device=self.device, dtype=self.dtype)


    def forward(self, x):
        # backbone
        feats = self.backbone(x)

        # reorg layer
        p5 = self.convsets_1(feats['layer3'])
        p4 = self.reorg(self.route_layer(feats['layer2']))
        p5 = torch.cat([p4, p5], dim=1)

        # head
        p5 = self.convsets_2(p5)

        # pred
        pred = self.pred(p5)
        return pred