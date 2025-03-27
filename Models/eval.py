import torch
import torch.nn.functional as F
from torcheval.metrics.aggregation.auc import AUC
from torchmetrics.detection import MeanAveragePrecision
import datetime


def raw_to_expected_outputsV2(out, num_classes, anchors):
    # input: (N, objects w/ relative coords, grid_size, grid_size)
    # single obj.: (objectness, box, classes) * num_boxes
    # [conf, obj_xc, obj_yc, obj_w, obj_h]
    obj_stride = num_classes+5
    out[:, 0::obj_stride, :, :] = out[:, 0::obj_stride, :, :].sigmoid() # objectness
    out[:, 1::obj_stride, :, :] = out[:, 1::obj_stride, :, :].sigmoid() # xc
    out[:, 2::obj_stride, :, :] = out[:, 2::obj_stride, :, :].sigmoid() # yc
    
    grid_size = out.shape[-1]
    _anchors = torch.tensor(anchors).to(out.device) * grid_size
    pw = _anchors[:, 0]
    ph = _anchors[:, 1]
    
    out[:, 3::obj_stride, :, :] = pw[None, :, None, None] * out[:, 3::obj_stride, :, :].exp() # w
    out[:, 4::obj_stride, :, :] = ph[None, :, None, None] * out[:, 4::obj_stride, :, :].exp() # h

    for i in range(len(anchors)):
        start_i = 5+i*obj_stride
        end_i = obj_stride*(i+1)
        out[:, start_i:end_i, :, :] = F.softmax(out[:, start_i:end_i, :, :], dim=1, dtype=out.dtype)
        
    # output: (N, objects w/ absolute coords, grid_size, grid_size)
    return out    

def get_absolute_boxesV2(out, num_classes, num_boxes):
    # bringing to outputs relative to the entire image
    grid_size = out.shape[-1]
    indexed_columns = torch.tensor([range(0,grid_size) for i in range(grid_size)], dtype=out.dtype, device=out.device)
    obj_stride = num_classes + 5
    out[:, 1::obj_stride, :, :] = (out[:, 1::obj_stride, :, :] + indexed_columns) / grid_size # xc
    out[:, 2::obj_stride, :, :] = (out[:, 2::obj_stride, :, :] + indexed_columns.T) / grid_size # yc
    
    out[:, 3::obj_stride, :, :] = out[:, 3::obj_stride, :, :] / grid_size # w
    out[:, 4::obj_stride, :, :] = out[:, 4::obj_stride, :, :] / grid_size # h
    return out

def reshape_for_eval(out, num_classes, num_boxes):
    batch_size = out.shape[0]
    obj_stride = num_classes + 5
    grid_size = out.shape[-1]
    out = out.view(batch_size, obj_stride*num_boxes, grid_size*grid_size)
    out = torch.concat(torch.split(out, obj_stride, 1), -1)
    out = out.permute(0,2,1) 
    
    # batch_size, number of objects, (classes + coords + objectness) * num_boxes
    return out

def sort_by_objectness(out):
    batch_size = out.shape[0]
    indeces = out[:, :, 0].argsort(descending=True)
    out = out[torch.arange(batch_size).unsqueeze(1), indeces]
    return out

def remove_below_threshold(out, obj_threshold):
    batch_size = out.shape[0]
    mask = out[:, :, 0] > obj_threshold
    return \
    [out[i, mask[i], :] for i in range(batch_size)]

def iou(box1, box2):
    # expects midpoint data
    xmin1 = box1[..., 0] - box1[..., 2] / 2 
    xmax1 = box1[..., 0] + box1[..., 2] / 2 
    ymin1 = box1[..., 1] - box1[..., 3] / 2
    ymax1 = box1[..., 1] + box1[..., 3] / 2

    xmin2 = box2[..., 0] - box2[..., 2] / 2 
    xmax2 = box2[..., 0] + box2[..., 2] / 2 
    ymin2 = box2[..., 1] - box2[..., 3] / 2
    ymax2 = box2[..., 1] + box2[..., 3] / 2

    xmin_i = torch.stack([xmin1, xmin2]).max(dim=0)[0]
    xmax_i = torch.stack([xmax1, xmax2]).min(dim=0)[0]
    ymin_i = torch.stack([ymin1, ymin2]).max(dim=0)[0]
    ymax_i = torch.stack([ymax1, ymax2]).min(dim=0)[0]

    intersection = F.relu(xmax_i-xmin_i) * F.relu(ymax_i-ymin_i)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    return intersection / (area1 + area2 - intersection + 1e-6)

def get_class_labels(eval_out):
    return \
        [torch.concat((img[:, :5], img[:, 5:].argmax(dim=1).unsqueeze(-1)), dim=-1) for img in eval_out]

def get_pred_boxes(out, anchors, num_classes=20, num_boxes=5, obj_threshold=0.5):
    out = raw_to_expected_outputsV2(out, num_classes, anchors)
    out = get_absolute_boxesV2(out, num_classes, num_boxes)
    out = reshape_for_eval(out, num_classes, num_boxes)
    out = sort_by_objectness(out)
    out = remove_below_threshold(out, obj_threshold)
    return get_class_labels(out)

def get_gt_boxes(gt_out, num_classes=20, num_boxes=5, obj_threshold=0.5):
    gt_out = get_absolute_boxesV2(gt_out, num_classes, num_boxes)
    gt_out = reshape_for_eval(gt_out, num_classes, num_boxes)
    gt_out = sort_by_objectness(gt_out)
    gt_out = remove_below_threshold(gt_out, obj_threshold)
    return get_class_labels(gt_out)

def NMS(pred_boxes, num_classes=20, iou_threshold=0.5):

    selected = []
    for img in pred_boxes:
        pred_class = img[:, 5]
        selected_for_img = []
        for cls in range(num_classes):
            # get objects of that class
            indeces = (pred_class == cls).nonzero(as_tuple=True)[0]
            if indeces.shape[0] < 1:
                continue
            objects = img[indeces]
            reference = objects[:1][:, 1:5] # highest IoU
            compared = objects[1:][:, 1:5]
            reference = reference.expand(compared.shape[0], 4)
            ious = iou(reference, compared)
            ious = torch.concat((torch.tensor([0.0], dtype=ious.dtype, device=ious.device), ious))
            selected_for_img.append(indeces[ious < iou_threshold])
        selected_for_img = torch.concat(selected_for_img) if selected_for_img else None
        selected.append(selected_for_img)

    return selected

def mAP(pred_boxes, gt_boxes, num_classes=20, iou_threshold=0.5):

    # create one tensor where 0 dim is number of objects and 1 dim is an object.
    # [[img_i, objectness_score, xc, yc, w, h, class_label, is_true_positive, precision, recall]]
    batch_size = len(gt_boxes)
    for img_i in range(batch_size):
        img_pred = pred_boxes[img_i]
        num_objects = img_pred.shape[0]
        img_i_column = torch.full((num_objects, 1), img_i, dtype=img_pred.dtype, device=img_pred.device)
        true_positives_column = torch.zeros(num_objects, dtype=img_pred.dtype, device=img_pred.device).unsqueeze(-1)
        metrics_columns = torch.full((num_objects, 2), -1, dtype=img_pred.dtype, device=img_pred.device)
        new_img_pred = torch.cat((img_i_column, img_pred, true_positives_column, metrics_columns), dim=-1)
        pred_boxes[img_i] = new_img_pred
    pred_boxes = torch.concat(pred_boxes)

    # figure out if predicted objects are true positives
    for pred_obj_i in range(pred_boxes.shape[0]):
        pred_obj = pred_boxes[pred_obj_i:pred_obj_i+1]
        img_i = int(pred_obj[0, 0])
        _class = pred_obj[0, 6]
        true_objs = gt_boxes[img_i][
                            (gt_boxes[img_i][:, 5] == _class).nonzero(as_tuple=True)[0]
        ]

        if true_objs.numel() == 0:
            continue
        pred_obj = pred_obj[:, 2:6]
        pred_obj = pred_obj.expand(true_objs.shape[0], 4)
        true_positive = torch.sum(
            iou(pred_obj, true_objs[:, 1:5]) > iou_threshold
        ).item() > 0
    
        if true_positive:
            pred_boxes[pred_obj_i, 7] = 1.0

    # sort by objectness score
    pred_boxes = pred_boxes[pred_boxes[:, 1].argsort(descending=True)]

    # calculate precision and recall for every class
    pred_class = pred_boxes[:, 6]
    results = {}
    for cls in range(num_classes):
        indeces_of_cls = (pred_class == cls).nonzero(as_tuple=True)[0]
        num_objects_of_cls = indeces_of_cls.shape[0]
        for obj_i in range(num_objects_of_cls):         
            objects = pred_boxes[indeces_of_cls[:obj_i+1]]
    
            # precision tp / (tp+fp)
            tp = objects[:, 7].sum()
            tp_plus_fp = objects.shape[0]
            precision = tp / tp_plus_fp
    
            # recall tp / (tp + fn)
            recall = tp / num_objects_of_cls

            pred_boxes[indeces_of_cls[obj_i], 8] = precision
            pred_boxes[indeces_of_cls[obj_i], 9] = recall

        if num_objects_of_cls > 0:
            metric = AUC()
            precision_scores = torch.cat([torch.tensor([1.], dtype=pred_boxes.dtype, device=pred_boxes.device), pred_boxes[indeces_of_cls, 8]])
            recall_scores = torch.cat([torch.tensor([0.], dtype=pred_boxes.dtype, device=pred_boxes.device), pred_boxes[indeces_of_cls, 9]])
            metric.update(recall_scores, precision_scores)
            ap_score = metric.compute()
            metric.reset()
            results[cls] = {'ap_score': ap_score, 'num_objects_of_cls': num_objects_of_cls}
        else:
            results[cls] = {'ap_score': -1, 'num_objects_of_cls': num_objects_of_cls}
            continue

    N = 0
    ap_sum = 0
    for cls, value in results.items():
        if value['num_objects_of_cls'] < 1:
            continue
        else:
            N += 1
            ap_sum += value['ap_score']

    results['mAP'] = ap_sum / N
    
    return results

def evaluate(model, loader, anchors, mAP=MeanAveragePrecision(box_format='cxcywh',class_metrics=True),
            num_classes=20, num_boxes=5, obj_threshold=0.5, iou_threshold=0.5):
    for i, (imgs, labels) in enumerate(loader):
        print(f'batch {i} {datetime.datetime.now()}')
        preds = []
        targets = []
        batch_size = labels.shape[0]
        with torch.no_grad():
            out = model(imgs).detach()
            pred_boxes = get_pred_boxes(out, anchors, num_classes=num_classes, num_boxes=num_boxes, obj_threshold=obj_threshold)
            gt_boxes = get_gt_boxes(labels.detach(), num_classes=num_classes, num_boxes=num_boxes, obj_threshold=obj_threshold)
            keep = NMS(pred_boxes, num_classes=num_classes, iou_threshold=iou_threshold)
            for j in range(batch_size):
                if keep[j] is None:
                    print(f'batch {i} image {j} contains no objects.')
                    continue
                boxes = pred_boxes[j][keep[j], ..., 1:5]
                scores = pred_boxes[j][keep[j], ..., 0]
                labels = pred_boxes[j][keep[j], ..., 5].to(torch.uint8)
                preds.append({'boxes':boxes,
                             'scores':scores,
                             'labels':labels})
    
                boxes = gt_boxes[j][..., 1:5]
                labels = gt_boxes[j][..., 5].to(torch.uint8)
                targets.append({'boxes':boxes,
                               'labels':labels})
    
        mAP.update(preds=preds, target=targets)
    result = mAP.compute()
    mAP.reset()
    return result