import torch
import torch.nn as nn
import datetime

class YOLOv2Loss(nn.Module):
    def __init__(self, anchors, lambda_noobj=0.5, lambda_coord=5.0, num_classes=20):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.softmax = torch.nn.Softmax(dim=2)
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord
        self.num_classes = num_classes
        self.anchors = anchors
        
    def forward(self, out, gt_out):
        # [conf, obj_xc, obj_yc, obj_w, obj_h]
        is_obj = gt_out[:, 0::25, ...] == 1.0
        no_obj = gt_out[:, 0::25, ...] == 0.0

        # CONFIDENCE LOSS ===========
        conf_true = gt_out[:, 0::25, ...]
        conf_pred = out[:, 0::25, ...].sigmoid()

        is_obj_conf_pred = is_obj * conf_pred
        is_obj_conf_true = is_obj * conf_true
        
        no_obj_conf_pred = no_obj * conf_pred
        no_obj_conf_true = no_obj * conf_true

        is_obj_conf_loss = self.mse(is_obj_conf_pred, is_obj_conf_true)
        no_obj_conf_loss = self.mse(no_obj_conf_pred, no_obj_conf_true) 
        # ===========================

        # BOX LOSS ==================
            # XCYC LOSS ==================
        xc_true = gt_out[:, 1::25, ...]
        yc_true = gt_out[:, 2::25, ...]

        xc_pred = out[:, 1::25, ...].sigmoid()
        yc_pred = out[:, 2::25, ...].sigmoid()

        xc_pred = is_obj * xc_pred
        xc_true = is_obj * xc_true
        yc_pred = is_obj * yc_pred
        yc_true = is_obj * yc_true

        xc_loss = self.mse(xc_pred, xc_true)
        yc_loss = self.mse(yc_pred, yc_true)
            # ============================

            # WH LOSS ====================
        
        w_true = gt_out[:, 3::25, ...]
        h_true = gt_out[:, 4::25, ...]
        
        scale = gt_out.shape[-1]
        _anchors = torch.tensor(self.anchors).to(out.device) * scale
        pw = _anchors[:, 0]
        ph = _anchors[:, 1]
        
        w_pred = out[:, 3::25, ...]
        h_pred = out[:, 4::25, ...]

        w_true = torch.log(
            1e-16 + w_true / pw[None, :, None, None]
        )
        h_true = torch.log(
            1e-16 + h_true / ph[None, :, None, None]
        )
        
        w_pred = is_obj * w_pred
        w_true = is_obj * w_true
        h_pred = is_obj * h_pred
        h_true = is_obj * h_true

        w_loss = self.mse(w_pred, w_true)
        h_loss = self.mse(h_pred, h_true)
        
            # ============================
        # ===========================
        
        # CLASS LOSS ================
        class_true = []
        for i in range(len(self.anchors)):
            first_idx = 5 + i*(5+self.num_classes)
            last_idx = 25 + i*(5+self.num_classes)
            class_true.append(gt_out[:, first_idx:last_idx, ...])
        class_true = torch.stack(class_true, dim=1)

        class_pred = []
        for i in range(len(self.anchors)):
            first_idx = 5 + i*(5+self.num_classes)
            last_idx = 25 + i*(5+self.num_classes)
            class_pred.append(gt_out[:, first_idx:last_idx, ...])
        class_pred = torch.stack(class_pred, dim=1)

        class_pred = self.softmax(class_pred)
        
        class_pred = is_obj[:, :, None, :, :] * class_pred
        class_true = is_obj[:, :, None, :, :] * class_true

        class_loss = self.mse(class_pred, class_true)
        # ===========================

        loss =  \
                self.lambda_coord * (w_loss + h_loss) + \
                self.lambda_coord * (xc_loss + yc_loss) + \
                is_obj_conf_loss + \
                self.lambda_noobj * no_obj_conf_loss + \
                class_loss
        return loss