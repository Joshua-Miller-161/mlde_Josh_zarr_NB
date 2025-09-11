import sys
sys.dont_write_bytecode = True
from torch.optim import Adam, SGD
#====================================================================
def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if (config.optim.optimizer == 'Adam'):
    optimizer = Adam(params,
                     lr=config.optim.lr,
                     betas=(config.optim.beta1, 0.999),
                     eps=config.optim.eps,
                     weight_decay=config.optim.weight_decay)
  
  elif (config.optim.optimizer == 'SGD'):
    optimizer = SGD(params,
                    lr=config.optim.lr,
                    momentum=0.9,
                    weight_decay=config.optim.weight_decay)
  
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer