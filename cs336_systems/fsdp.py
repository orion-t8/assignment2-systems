import torch
import torch.distributed as dist
import torch.nn as nn

class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.handles = []
        self.sharded_params = set()
        self.registry = dict() # sharded module -> sharded param (kept fp32 all the time)
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        def gather_full_weight(mod: nn.Module, input):
            shard_weight = self.registry[mod]
            if compute_dtype is not None:
                shard_weight = shard_weight.to(compute_dtype)
            full_weight_list = [torch.empty_like(shard_weight) for _ in range(world_size)]
            dist.all_gather(full_weight_list, shard_weight)
            mod.weight.data = torch.cat(full_weight_list, dim=0)

        def swap_back_shard_weight(mod: nn.Module, input, output):
            mod.weight.data = self.registry[mod]

        def reduce_scatter_shard_gradient(mod):
            def hook(param: nn.Parameter):
                param.data = self.registry[mod]
                shard_size = param.grad.shape[0] // world_size
                grad_list = [param.grad[i*shard_size : (i+1)*shard_size, ...]/world_size for i in range(world_size)] # compute_dtype
                shard_grad = torch.empty((shard_size,) + param.grad.shape[1:], dtype=compute_dtype)
                handle = dist.reduce_scatter(shard_grad, grad_list, async_op=True)
                # 存grad_list避免通信线程读取grad_list完成之前，grad_list已被回收
                # 存shard_grad是因为要等异步reduce_scatter完成以后再把grad cast回原精度
                self.handles.append((handle, grad_list, shard_grad, param))
            return hook

        def all_reduce_unsharded_gradient(param: nn.Parameter):
            param.grad /= world_size
            handle = dist.all_reduce(param.grad, async_op=True)
            self.handles.append((handle,)) # 存成tuple为了和reduce_scatter的场景保持一致

        from cs336_basics.model import Embedding, Linear
        for mod in self.module.modules():
            if not isinstance(mod, (Linear, Embedding)):
                if hasattr(mod, "weight") and mod.weight is not None:
                    mod.weight.register_post_accumulate_grad_hook(all_reduce_unsharded_gradient)
                continue
            self.sharded_params.add(mod.weight)
            shard_size = mod.weight.data.shape[0] // world_size
            shard_data = mod.weight.data[rank*shard_size : (rank+1)*shard_size, ...].clone()
            mod.weight.data = shard_data
            self.registry[mod] = shard_data
            mod.register_forward_pre_hook(gather_full_weight)
            mod.register_forward_hook(swap_back_shard_weight)
            mod.register_full_backward_pre_hook(gather_full_weight)
            mod.weight.register_post_accumulate_grad_hook(reduce_scatter_shard_gradient(mod))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for meta_info in self.handles:
            if len(meta_info) == 1:
                meta_info[0].wait()
            if len(meta_info) == 4:
                handle, _, grad, param = meta_info    
                handle.wait()
                param.grad = grad.to(param.data.dtype)
        self.handles.clear()