import torch
import torch._utils
import torch.distributed as dist
import torch.nn as nn

class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        for value in self.module.state_dict().values():
            dist.broadcast(value, src=0)

    def forward(self, ddp_data):
        return self.module(ddp_data)

    def finish_gradient_synchronization(self):
        grad_list = []
        for p in self.module.parameters():
            if p.grad is None:
                continue
            grad_list.append(p.grad.data / dist.get_world_size())
        flattened_grads = torch._utils._flatten_dense_tensors(grad_list)
        dist.all_reduce(flattened_grads)
        unflattened_grads = torch._utils._unflatten_dense_tensors(flattened_grads, grad_list)
        i = 0
        for p in self.module.parameters():
            if p.grad is None:
                continue
            p.grad.data = unflattened_grads[i]
            i += 1


class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = [] # store gradient all-reduce handles
        for value in self.module.state_dict().values():
            dist.broadcast(value, src=0)
        def all_reduce_grad(p: nn.parameter.Parameter):
            if p.grad.data is not None:
                p.grad.data /= dist.get_world_size()
                self.handles.append(dist.all_reduce(p.grad.data, async_op=True))
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(all_reduce_grad)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()