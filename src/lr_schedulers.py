import sys
sys.dont_write_bytecode = True
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
#====================================================================
class FixedLR(_LRScheduler):
    def __init__(self, optimizer, config, last_epoch=-1):
        self.lr = config.optim.lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lr for _ in self.optimizer.param_groups]
#====================================================================
class NoisyDecayLR(_LRScheduler):
    def __init__(self, optimizer, config, mag_noise=1, last_epoch=-1):
        self.initial_lr   = config.optim.lr
        self.final_lr     = config.optim.final_lr
        self.total_epochs = config.training.n_epochs
        self.mag_noise    = mag_noise

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        decay_rate = self.final_lr / self.initial_lr
        decayed_lr = self.initial_lr * (decay_rate ** (epoch / self.total_epochs))
        
        noise = np.random.uniform(-1, 1) * self.mag_noise * np.sqrt(decayed_lr * self.initial_lr)
        noisy_lr = decayed_lr + noise

        if noisy_lr < self.final_lr:
            noisy_lr = self.final_lr

        return [noisy_lr for _ in self.optimizer.param_groups]
#====================================================================
class TriangleFractalLR(_LRScheduler):
    def __init__(self, optimizer, config, last_epoch=-1):
        self.init_lr      = config.optim.lr
        self.final_lr     = config.optim.final_lr
        self.total_epochs = config.training.n_epochs
        self.num_waves    = config.optim.num_waves
        self.period       = config.optim.period

        super().__init__(optimizer, last_epoch)

    # def MajorPeakHeight(self, epoch):
    #     n = int(epoch / (self.num_waves * self.period))
    #     m = - (1 / ((n + 1) * (n + 2))) * (self.final_lr + self.init_lr) / (self.num_waves * self.period)
    #     b = (self.final_lr + self.init_lr) / (n + 1) + (n * (self.final_lr + self.init_lr)) / ((n + 1) * (n + 2))
    #     x = (self.num_waves * self.period) * int(epoch / (self.num_waves * self.period))
    #     return m * x + b
    
    # def SubPeakHeight(self, epoch, top):
    #     m = (self.final_lr - top) / (self.period * self.num_waves)
    #     y_curr = m * epoch + top
    #     y_peak = m * (self.period * int(epoch / self.period)) + top
    #     return max(y_curr, y_peak)

    # def WaveLine(self, epoch, peak):
    #     m = (self.final_lr - peak) / self.period
    #     b = peak - m * self.period * (int(epoch / self.period))
    #     return m * epoch + b
    
    def major_peak_height(self, epoch):
        m = (self.final_lr - self.init_lr) / self.total_epochs
        x = (self.num_waves * self.period) * int(epoch / (self.num_waves * self.period))
        return m * x + self.init_lr

    def sub_peak_height(self, epoch, top):
        m = (self.final_lr - top) / (self.period * self.num_waves)
        y_curr = m * epoch + top
        y_peak = m * (self.period * int(epoch / self.period)) + top
        return max(y_curr, y_peak)

    def wave_line(self, epoch, peak):
        m = (self.final_lr - peak) / self.period
        b = peak - m * self.period * (int(epoch / self.period))
        return m * epoch + b
    
    def get_lr(self):
        epoch = self.last_epoch + 1
        major = self.major_peak_height(epoch)
        peak_lr = self.sub_peak_height(epoch % (self.num_waves * self.period), major)
        lr = self.wave_line(epoch, peak_lr)
        return [lr for _ in self.optimizer.param_groups]
#====================================================================
def get_lr_scheduler(optimizer, config):

    print(" v v v v v v v v v v v v v v v v v v v v v v v v v v v v")
    print(" >> >> inside get_lr_scheduler")
    print(config.to_dict())
    print(" ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^")

    if (config.optim.scheduler == 'fixed'):
        return FixedLR(optimizer, config)
    elif (config.optim.scheduler == 'decay'):
        return NoisyDecayLR(optimizer, config)
    elif (config.optim.scheduler == 'triangle'):
        return TriangleFractalLR(optimizer, config)