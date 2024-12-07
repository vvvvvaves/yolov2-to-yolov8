import torch
from torch.profiler import profile, ProfilerActivity
import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy(history):
    plt.plot(history['train_accuracy'], label='train accuracy')
    plt.plot(history['val_accuracy'], label='val accuracy')
    plt.legend()
    return plt

def show_tree(obj, r=0, n=90):
    if r == 0:
        obj = obj.key_averages()[0]
    children = obj.cpu_children
    if 'name' in dir(obj):
        name = ' '*3*r+obj.name
        offset = ' '*(n - len(name))
        
        tr = obj.time_range
        print(name + offset + str(tr.start) + ' ' + str(tr.end))
    else:
        print('\t'*r+obj.key)
    if len(children) > 0:
        for child in children:
            show_tree(child, r=r+1, n=n)

def generate_dataframe(prof):

    def get_cpu_children(obj, attrs, _rows, r=0):
        row = {}
        for attr in attrs:
            if attr in dir(obj):
                row[attr] = getattr(obj, attr)
            else:
                row[attr] = None
        row['r'] = r
        _rows.append(row)
        children = row['cpu_children']
        if len(children) > 0:
            for child in children:
                _rows = get_cpu_children(child, attrs, _rows, r=r+1)
        return _rows
    
    step = prof.key_averages()[0]
    _attrs = dir(prof.key_averages()[0].cpu_children[0])
    _attrs = [child for child in _attrs if child[:2] != '__' 
                   and child not in ['set_cpu_parent', 'append_cpu_child', 'append_kernel', 'set_cpu_parent']]
    _rows = []
    _rows = get_cpu_children(step, _attrs, _rows)

    for row in _rows:
        row['cpu_children'] = [child.id for child in row['cpu_children']]
        row['cpu_parent'] = None if row['cpu_parent'] is None else row['cpu_parent'].id
        row['tr.start'] =  None if row['time_range'] is None else row['time_range'].start
        row['tr.end'] = None if row['time_range'] is None else row['time_range'].end
        row.pop('time_range', None)
        
    df = pd.DataFrame(_rows)
    return df

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())