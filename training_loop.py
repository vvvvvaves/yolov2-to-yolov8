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
        print(f"{_datetime} Batch {i+1}: ")
        outputs = model(imgs)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_train += loss.item()
        preds = outputs.max(1)[1]
        correct += preds.eq(labels).sum().item()
        total += labels.shape[0]

    n_batches = len(train_loader)
    accuracy = round(correct / total * 100., 4)
    loss_per_batch = round(loss_train / n_batches * 100., 4)
    print(f'[Train] Accuracy: {accuracy}, Loss per batch: {loss_per_batch}')
    return accuracy, loss_per_batch