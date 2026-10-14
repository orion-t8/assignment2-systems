import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import numpy as np
import pandas as pd

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def run_all_reduce(rank, world_size, data_size, queue: mp.Queue):
    setup(rank, world_size)
    data = torch.rand((data_size,), dtype=torch.float32, device="cuda")
    for _ in range(5):
        # warmup steps
        torch.cuda.synchronize()
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()
    dist.destroy_process_group()

    time_over_steps = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = timeit.default_timer()
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        time_over_ranks = [None for _ in range(world_size)]
        dist.all_gather_object(time_over_ranks, end-start)
        time_over_steps.append(np.max(time_over_ranks))
    if rank == 0:
        metric = {
            "world size": world_size,
            "data size": data_size,
            "median over max over ranks": np.median(time_over_steps)
        }
        queue.put(metric)
        print(f"world_size: {world_size}, data_size: {data_size}, median: {np.median(time_over_steps)}")

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    shared_queue = ctx.Queue()
    for world_size in [2, 4, 6]:
        for j in [1, 10, 100, 1024]:
            data_size = j * 1000000 // 4
            mp.spawn(fn=run_all_reduce, args=(world_size, data_size, shared_queue), nprocs=world_size, join=True)
    shared_queue.put(None)
    collected_data = []
    while True:
        data = shared_queue.get()
        if data is None:
            break
        collected_data.append(data)
    if collected_data:
        df = pd.DataFrame(collected_data)
        df.to_csv("./all_reduce_benchmark.csv", index=None)