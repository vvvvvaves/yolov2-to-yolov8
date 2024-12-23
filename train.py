import os
import pickle
import datetime
from loop import training_loop, validation_loop
import pandas as pd
import torch
from torch.autograd.profiler import record_function

from cls_train import get_gradient_stats, save_train_outputs, load_train_outputs, load_model_with_grads

def train(epochs, train_loader, val_loader, model, optimizer, 
          loss_fn, scheduler, scaler, outputs_path, save_at=1, save_grad=True, 
          resume=False):
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
        scaler.load_state_dict(state['scaler'])
    else:
        starting_epoch = 1
        history = pd.DataFrame(columns=['datetime',
                                       'epoch',
                                       'train_mAP',
                                       'train_loss_per_batch',
                                       'val_mAP',
                                       'val_loss_per_batch'])
    
        gradient_stats = []
        
    for epoch in range(starting_epoch, int(epochs + starting_epoch)):
        _datetime = datetime.datetime.now()
        print(f"{_datetime} Epoch {epoch} ")
        train_map, train_loss = training_loop(optimizer, model, loss_fn, train_loader, scaler)

        val_map, val_loss = validation_loop(model, val_loader, loss_fn)
        _gradient_stats = get_gradient_stats(model)

        history.loc[epoch - 1] = [_datetime, epoch, train_map, train_loss, val_map, val_loss]
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
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict()
           }
    torch.save(state, os.path.join(outputs_path, f"state.pt"))
    save_train_outputs(outputs_path, history, gradient_stats)
    torch.save(model, os.path.join(model_path, f"model_final.pt"))

    return history, gradient_stats