import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from typing import Type, Any

class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.sharded_param_groups = []
        self._inner_optimizer = None
        self.counter = 0 # count num_params
        super().__init__(params, kwargs)
        if len(self.sharded_param_groups) > 0:
            self._inner_optimizer = optimizer_cls(self.sharded_param_groups, **kwargs)

    def step(self, closure=None, **kwargs):
        ret = self._inner_optimizer.step(closure, **kwargs)
        counter = 0
        for param_group in self.param_groups:
            for param in param_group['params']:
                dist.broadcast(param, src=counter % dist.get_world_size())
                counter += 1
        return ret

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        sharded_param_group = dict()
        sharded_param_group['params'] = []
        for param in param_group['params']:
            if self.counter % dist.get_world_size() == dist.get_rank():
                sharded_param_group['params'].append(param)
            self.counter += 1
        if len(sharded_param_group['params']) == 0:
            return
        for key, value in param_group.items():
            if key != 'params':
                sharded_param_group[key] = value
        self.sharded_param_groups.append(sharded_param_group)
        if self._inner_optimizer:
            self._inner_optimizer.add_param_group(sharded_param_group)