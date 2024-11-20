import thop
from copy import deepcopy
import torch.nn as nn
import torch
""" in get_flops I changed up the function to give an option of half precision """

TORCH_2_0 = True
def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model

def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))

def get_flops(model, imgsz=640, half=False): # Changed up the function to give an option of half precision
    """Return a YOLO model's FLOPs."""
    if not thop:
        print('here')
        return 0.0  # if not installed return 0.0 GFLOPs

    model = de_parallel(model)
    p = next(model.parameters())
    if not isinstance(imgsz, list):
        imgsz = [imgsz, imgsz]  # expand if int/float
    try:
        # Use stride size for input tensor
        stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # max stride
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device) # input image in BCHW format
        if half:
            im = im.half()
        flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # stride GFLOPs
        return flops * imgsz[0] / stride * imgsz[1] / stride  # imgsz GFLOPs
    except Exception:
        # Use actual image size for input tensor (i.e. required for RTDETR models)
        im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
        return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # imgsz GFLOPs

def get_flops_with_torch_profiler(model, imgsz=640):
    """Compute model FLOPs (thop package alternative, but 2-10x slower unfortunately)."""
    if not TORCH_2_0:  # torch profiler implemented in torch>=2.0
        return 0.0
    model = de_parallel(model)
    p = next(model.parameters())
    if not isinstance(imgsz, list):
        imgsz = [imgsz, imgsz]  # expand if int/float
    try:
        # Use stride size for input tensor
        stride = (max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32) * 2  # max stride
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum(x.flops for x in prof.key_averages()) / 1e9
        flops = flops * imgsz[0] / stride * imgsz[1] / stride  # 640x640 GFLOPs
    except Exception:
        # Use actual image size for input tensor (i.e. required for RTDETR models)
        im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum(x.flops for x in prof.key_averages()) / 1e9
    return flops