def IoU(box1, box2, midpoint=True):
    if midpoint:
        x1 = box1[0]
        y1 = box1[1]
        w1 = box1[2]
        h1 = box1[3]
    
        x2 = box2[0]
        y2 = box2[1]
        w2 = box2[2]
        h2 = box2[3]
    
        xmin1 = x1 - w1/2
        xmin2 = x2 - w2/2
        ymin1 = y1 - h1/2
        ymin2 = y2 - h2/2
    
        xmax1 = x1 + w1/2
        xmax2 = x2 + w2/2
        ymax1 = y1 + h1/2
        ymax2 = y2 + h2/2
    else:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    
    xmin_i = max(xmin1, xmin2)
    xmax_i = min(xmax1, xmax2)
    ymin_i = max(ymin1, ymin2)
    ymax_i = min(ymax1, ymax2)

    intersection = max(xmax_i-xmin_i, 0) * max(ymax_i-ymin_i, 0)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    return intersection / (area1 + area2 - intersection + 1e-6)