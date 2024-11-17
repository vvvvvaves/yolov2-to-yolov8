import os
import json
import datetime
from training_loop import training_loop
from validation_loop import validation_loop
import pandas as pd
import torch
from torch.autograd.profiler import record_function

def get_gradient_stats(model):
    stats = {'parameter_i': [], 'mean': [], 'median': [], 'std': [], 'size_of_tensor':[]}
    for i, parameter in enumerate(model.parameters()):
        stats['parameter_i'].append(i)
        _grad = parameter.grad.flatten()
        stats['mean'].append(torch.mean(_grad))
        stats['median'].append(torch.median(_grad))
        stats['std'].append(torch.std(_grad))
        stats['size_of_tensor'].append(len(_grad))
    return pd.DataFrame(stats)

def save_train_outputs(path, history, gradient_stats, prof):
    _dict = {'history': history,
             'gradient_stats': gradient_stats,
             'prof': prof}
    
    with open(os.path.join(path, 'train_outputs.json'), 'wb') as file:
        json.dump(_dict, file)

def load_train_outputs(path):
    with open(os.path.join(path, 'train_outputs.json'), 'rb') as file:
        return json.load(file)

def train(epochs, train_loader, val_loader, model, optimizer, loss_fn, profiler):

    history = pd.DataFrame(columns=['datetime',
                                   'epoch',
                                   'train_accuracy',
                                   'train_loss_per_batch',
                                   'val_accuracy',
                                   'val_loss_per_batch'])

    gradient_stats = []
    profiler.start()
    for epoch in range(1, int(epochs) + 1):
        profiler.step()
        _datetime = datetime.datetime.now()
        print(f"{_datetime} Epoch {epoch} ")
        with record_function("Training Loop"):
            train_accuracy, train_loss = training_loop(optimizer, model, loss_fn, train_loader)
        with record_function("Validation Loop"):
            val_accuracy, val_loss = validation_loop(model, val_loader, loss_fn)
        _gradient_stats = get_gradient_stats(model)

        history.loc[epoch - 1] = [_datetime, epoch, train_accuracy, train_loss, val_accuracy, val_loss]
        gradient_stats.append(_gradient_stats)

    #    result = {'stats': [_datetime, epoch] + stats,
    #              'model': model.state_dict()}

    #    torch.save(result, f"./result_e{epoch}.pt")
    profiler.stop()
    return history, gradient_stats, profiler