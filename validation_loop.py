import torch
import datetime
from torch.utils.data import DataLoader
from torch import nn
from torch.profiler import record_function

def validation_loop(model,
                    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
                    loss_fn: nn.Module) -> list[float]:

    model.eval()

    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(val_loader):
            imgs = imgs.to('cuda:0')
            labels = labels.to('cuda:0')
            
            if (i+1) % 15 == 0:
                _datetime = datetime.datetime.now()
                print(f"{_datetime} Batch {i+1} ")

            outputs = model(imgs)
            preds = outputs.max(1)[1]

            total += labels.shape[0]
            correct += preds.eq(labels).sum().item()

            loss += loss_fn(outputs, labels).item()

    accuracy = round(correct / total * 100., 4)
    loss_per_batch = round(loss / len(val_loader), 4)
    print(f"[Val] Accuracy: {accuracy}%,"
          + f" loss per batch: {loss_per_batch}")
    return accuracy, loss_per_batch