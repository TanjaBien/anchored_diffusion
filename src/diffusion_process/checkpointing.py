import os
from pathlib import Path

import torch
import yaml


class EMA:
    """Exponential moving average of the trainable weights, swapped in for evaluation."""

    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}
        self.backup = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.mu).add_(param.data, alpha=1.0 - self.mu)

    def ema(self, module, backup=True):
        for name, param in module.named_parameters():
            if param.requires_grad:
                if backup:
                    self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


def save_model(config, model, train_iterations, output_save_dir):
    os.makedirs(Path(output_save_dir, 'model/'), exist_ok=True)

    with open(output_save_dir + '/model/model.yaml', 'w') as yaml_file:
        yaml.dump(dict(config), yaml_file, default_flow_style=False)

    save_dir_model = output_save_dir + '/model/checkpoint_' + str(train_iterations) + '.pt'
    save_obj = {'model': model.state_dict()}
    with open(save_dir_model, 'wb') as f:
        torch.save(save_obj, f)
    print(f'\ncheckpoint saved to {output_save_dir}/.')


def load_model(path, model, strict=True):
    with open(path, 'rb') as f:
        loaded_obj = torch.load(f, map_location='cpu')
    model.load_state_dict(loaded_obj['model'], strict=strict)
    print(f'\nCheckpoint loaded from {path}')
    return model
