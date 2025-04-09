from typing import Union, List

import torch
import torch.nn as nn
from typing import Callable
from src.diffusion.base.scheduling import BaseScheduler

class BaseSampler(nn.Module):
    def __init__(self,
                 scheduler: BaseScheduler = None,
                 guidance_fn: Callable = None,
                 num_steps: int = 250,
                 guidance: Union[float, List[float]] = 1.0,
                 *args,
                 **kwargs
        ):
        super(BaseSampler, self).__init__()
        self.num_steps = num_steps
        self.guidance = guidance
        self.guidance_fn = guidance_fn
        self.scheduler = scheduler

    
    def _impl_sampling(self, net, noise, condition, uncondition):
        raise NotImplementedError

    def __call__(self, net, noise, condition, uncondition):
        denoised = self._impl_sampling(net, noise, condition, uncondition)
        return denoised


