import torch
import datetime
from torch.utils.data import DataLoader

def training_loop(optimizer: torch.optim.Optimizer,
                  model,
                  loss_fn: torch.nn.Module,
                  train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
                  scaler) -> None:
    model.train()
    loss_train = 0.0
    for i, (imgs, labels) in enumerate(train_loader):
        if (i+1) % 30 == 0:
            _datetime = datetime.datetime.now()
            print(f"{_datetime} Batch {i+1} ")

        # ====================
        # SCALER

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            optimizer.zero_grad()

            out = model(imgs)

            loss = loss_fn(out, labels)
    
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        scaler.step(optimizer)

        scaler.update()

        # ==========================
        # NO SCALER

        # outputs = model(imgs)

        # loss = loss_fn(outputs, labels)
        
        # optimizer.zero_grad()

        # loss.backward()

        # optimizer.step()

        #=============================

        loss_train += loss.item()

        del imgs, labels, out

    n_batches = len(train_loader)
    loss_per_batch = round(loss_train / n_batches, 4)
    print(f'[Train] Loss per batch: {loss_per_batch}')
    return loss_per_batch

def validation_loop(model,
                    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
                    loss_fn: torch.nn.Module) -> list[float]:

    model.eval()
    loss_val = 0.0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(val_loader):
            
            if (i+1) % 30 == 0:
                _datetime = datetime.datetime.now()
                print(f"{_datetime} Batch {i+1} ")

            with torch.amp.autocast("cuda"):
                out = model(imgs)
                loss = loss_fn(out, labels)
            loss_val += loss.item()
            del imgs, labels, out

    loss_per_batch = round(loss_val / len(val_loader), 4)
    print(f"[Val]"
          + f" loss per batch: {loss_per_batch}")
    return loss_per_batch