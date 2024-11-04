import datetime
from training_loop import training_loop
from validation_loop import validation_loop
import pandas as pd
import torch


def train(epochs, train_loader, val_loader, model, optimizer, loss_fn):

    history = pd.DataFrame(columns=['datetime',
                                   'epoch',
                                   'train_accuracy',
                                   'train_loss_per_batch',
                                   'val_accuracy',
                                   'val_loss_per_batch'])

    for epoch in range(1, int(epochs) + 1):
        _datetime = datetime.datetime.now()
        print(f"{_datetime} Epoch {epoch}: ")
        train_accuracy, train_loss = training_loop(optimizer, model, loss_fn, train_loader)
        val_accuracy, val_loss = validation_loop(model, val_loader, loss_fn)

        history.loc[epoch - 1] = [_datetime, epoch, train_accuracy, train_loss, val_accuracy, val_loss]

    #    result = {'stats': [_datetime, epoch] + stats,
    #              'model': model.state_dict()}

    #    torch.save(result, f"./result_e{epoch}.pt")

    return history

