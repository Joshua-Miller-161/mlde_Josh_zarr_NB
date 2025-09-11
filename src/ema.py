# Modified from https://raw.githubusercontent.com/fadel/pytorch_ema/master/torch_ema/ema.py

from __future__ import division
from __future__ import unicode_literals

import torch

# Partially based on: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py

class ExponentialMovingAverage:
    def __init__(self, parameters, decay, use_num_updates=True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach().to(p.device) for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        if self.decay == 1:
            self.shadow_params = [p.clone().detach().to(p.device) for p in parameters if p.requires_grad]
        else:
            decay = self.decay
            if self.num_updates is not None:
                self.num_updates += 1
                decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
            one_minus_decay = 1.0 - decay
            with torch.no_grad():
                parameters = [p for p in parameters if p.requires_grad]
                for s_param, param in zip(self.shadow_params, parameters):
                    if s_param.device != param.device:
                        s_param.data = s_param.data.to(param.device)
                    s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if s_param.device != param.device:
                s_param.data = s_param.data.to(param.device)
            param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [p.clone().detach().to(p.device) for p in parameters]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            if c_param.device != param.device:
                c_param = c_param.to(param.device)
            param.data.copy_(c_param.data)

    def state_dict(self):
        return {
            'decay': self.decay,
            'num_updates': self.num_updates,
            'shadow_params': [p.cpu() for p in self.shadow_params]
        }

    def load_state_dict(self, state_dict, device=None):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [
            p.to(device) if device is not None else p
            for p in state_dict['shadow_params']
        ]