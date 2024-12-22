import torch
import datetime
from torch.utils.data import DataLoader

def training_loop(optimizer: torch.optim.Optimizer,
                  model,
                  loss_fn: torch.nn.Module,
                  train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> None:
    model.train()
    loss_train = 0.0
    correct = 0
    total = 0
    for i, (imgs, labels) in enumerate(train_loader):
        _datetime = datetime.datetime.now()
        print(f"{_datetime} After batch load / before to(cuda)")
        imgs = imgs.to('cuda:0')
        labels = labels.to('cuda:0')
        _datetime = datetime.datetime.now()
        print(f"{_datetime} After to(cuda)")
        
        if (i+1) % 1 == 0:
            _datetime = datetime.datetime.now()
            print(f"{_datetime} Batch {i+1} ")

        _datetime = datetime.datetime.now()
        print(f"{_datetime} Before inference on 64 batch")
        outputs = model(imgs)
        _datetime = datetime.datetime.now()
        print(f"{_datetime} After inference on 64 batch / before loss fn")
        loss = loss_fn(outputs, labels)
        _datetime = datetime.datetime.now()
        print(f"{_datetime} After loss fn / before zero grad")
        
        optimizer.zero_grad()
        _datetime = datetime.datetime.now()
        print(f"{_datetime} After zero grad / before backward")

        loss.backward()
        _datetime = datetime.datetime.now()
        print(f"{_datetime} After backward / before optimizer.step")

        optimizer.step()
        _datetime = datetime.datetime.now()
        print(f"{_datetime} After optimizer.step / before += loss.item()")

        loss_train += loss.item()
        _datetime = datetime.datetime.now()
        print(f"{_datetime} After += loss.item() / before the next batch")
        
        # preds = outputs.max(1)[1]
        # correct += preds.eq(labels).sum().item()
        # total += labels.shape[0]

    n_batches = len(train_loader)
    # accuracy = round(correct / total * 100., 4)
    loss_per_batch = round(loss_train / n_batches, 4)
    print(f'[Train] Loss per batch: {loss_per_batch}')
    return 0, loss_per_batch

def validation_loop(model,
                    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
                    loss_fn: torch.nn.Module) -> list[float]:

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
            # preds = outputs.max(1)[1]

            # total += labels.shape[0]
            # correct += preds.eq(labels).sum().item()

            loss += loss_fn(outputs, labels).item()

    # accuracy = round(correct / total * 100., 4)
    loss_per_batch = round(loss / len(val_loader), 4)
    print(f"[Val]"
          + f" loss per batch: {loss_per_batch}")
    return 0, loss_per_batch