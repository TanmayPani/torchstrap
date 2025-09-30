import torch

from .grad_transform import OptimState
    




class LRScheduler:
    def __init__(
        self, 
        optim_state : OptimState,
        last_epoch: int = -1,
    ) -> None :
        self.optimizer_state = optim_state
        self.last_epoch = last_epoch


