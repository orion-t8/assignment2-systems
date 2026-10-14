import torch
import torch.distributed as dist
import torch.nn as nn

class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        for value in self.module.state_dict().values():
            dist.broadcast(value, src=0)

    def forward(self, ddp_data):
        return self.module(ddp_data)

    def do_gradient_sync(self):
        for p in self.module.parameters():
            if p.grad is None:
                continue
            grad = p.grad.data / dist.get_world_size()
            dist.all_reduce(grad)
            p.grad.data = grad