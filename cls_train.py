import os
import pickle
import datetime
from cls_loop import training_loop, validation_loop
import pandas as pd
import torch
from torch.autograd.profiler import record_function

def get_gradient_stats(model):
    stats = {'parameter_i': [], 'name': [], 'mean': [], 'median': [], 'std': [], 'max': [], 'min': [], 'mean_abs': [], 'median_abs': [], 'std_abs': [], 'size_of_tensor':[]}
    for i, (name, parameter) in enumerate(model.named_parameters()):
        stats['parameter_i'].append(i)
        stats['name'].append(name)
        _grad = parameter.grad.flatten()
        stats['mean'].append(torch.mean(_grad).item())
        stats['median'].append(torch.median(_grad).item())
        stats['std'].append(torch.std(_grad).item())
        stats['max'].append(torch.max(_grad).item())
        stats['min'].append(torch.min(_grad).item())
        stats['mean_abs'].append(torch.mean(torch.abs(_grad)).item())
        stats['median_abs'].append(torch.median(torch.abs(_grad)).item())
        stats['std_abs'].append(torch.std(torch.abs(_grad)).item())
        stats['size_of_tensor'].append(len(_grad))
    return pd.DataFrame(stats)

def save_train_outputs(path, history, gradient_stats):
    _dict = {'history': history,
             'gradient_stats': gradient_stats}
    
    with open(os.path.join(path, 'train_outputs.pickle'), 'wb') as file:
        pickle.dump(_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_train_outputs(path):
    with open(os.path.join(path, 'train_outputs.pickle'), 'rb') as file:
        return pickle.load(file)

def load_model_with_grads(model_path, epoch):
    model = torch.load(os.path.join(model_path, f"model_e{epoch}.pt"))
    grad_dict = torch.load(os.path.join(model_path, f"grad_e{epoch}.pt"))
    for name, p in model.named_parameters():
        p.grad = grad_dict[name]
    return model

def train(epochs, train_loader, val_loader, model, optimizer, loss_fn, scheduler, outputs_path, save_at=1, save_grad=True, resume=False):
    model_path = os.path.join(outputs_path, 'models/')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if resume:
        outputs = load_train_outputs(outputs_path)
        history = outputs['history']
        gradient_stats = outputs['gradient_stats']
        state = torch.load(os.path.join(outputs_path, f"state.pt"))
        starting_epoch = state['epoch']
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
    else:
        starting_epoch = 1
        history = pd.DataFrame(columns=['datetime',
                                       'epoch',
                                       'train_accuracy',
                                       'train_loss_per_batch',
                                       'val_accuracy',
                                       'val_loss_per_batch'])
    
        gradient_stats = []
        
    for epoch in range(starting_epoch, int(epochs + starting_epoch)):
        _datetime = datetime.datetime.now()
        print(f"{_datetime} Epoch {epoch} ")
        train_accuracy, train_loss = training_loop(optimizer, model, loss_fn, train_loader)

        val_accuracy, val_loss = validation_loop(model, val_loader, loss_fn)
        _gradient_stats = get_gradient_stats(model)

        history.loc[epoch - 1] = [_datetime, epoch, train_accuracy, train_loss, val_accuracy, val_loss]
        gradient_stats.append(_gradient_stats)

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))

        if epoch % save_at == 0:
            torch.save(model, os.path.join(model_path, f"model_e{epoch}.pt"))
            if save_grad:
                grad_dict = {x[0]:x[1].grad for x in model.named_parameters()}
                torch.save(grad_dict, os.path.join(model_path, f"grad_e{epoch}.pt"))

    
    state = {'epoch': int(epochs + starting_epoch),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
           }
    torch.save(state, os.path.join(outputs_path, f"state.pt"))
    save_train_outputs(outputs_path, history, gradient_stats)
    torch.save(model, os.path.join(model_path, f"model_final.pt"))

    return history, gradient_stats