from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import hydra
import random
import numpy as np
import torch
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import timeit
import pandas as pd
from datetime import datetime
from contextlib import nullcontext
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    assert cfg.training.device.startswith("cuda")
    random.seed(cfg.training.seed) 
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    if cfg.training.device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.training.seed)

    dataset = np.random.randint(0, cfg.model.vocab_size, cfg.model.context_length*10)
    x, y = get_batch(dataset, cfg.training.batch_size, cfg.model.context_length, cfg.training.device)
    model = BasicsTransformerLM(cfg.model.vocab_size, cfg.model.context_length, cfg.model.d_model,
                               cfg.model.num_layers, cfg.model.num_heads, cfg.model.d_ff, cfg.model.rope_theta)
    model.to(cfg.training.device)
    optimizer = AdamW(model.parameters())
    mode = cfg.benchmark.mode
    assert mode in ["forward-only", "forward-and-backward", "full"]
    res = []
    for t in range(cfg.benchmark.warmup_steps + cfg.benchmark.measure_steps):
        specs = {
            "size": cfg.size,
            "d_model": cfg.model.d_model,
            "d_ff": cfg.model.d_ff,
            "num_layers": cfg.model.num_layers,
            "num_heads": cfg.model.num_heads,
            "context_length": cfg.model.context_length,
            "batch_size": cfg.training.batch_size,
            "warmup_steps": cfg.benchmark.warmup_steps,
            "measurement_steps": cfg.benchmark.measure_steps,
            "mode": cfg.benchmark.mode
        }
        optimizer.zero_grad()
        is_measurement = t >= cfg.benchmark.warmup_steps
        torch.cuda.synchronize()

        logits = None
        start = timeit.default_timer()

        forward_context = torch.no_grad() if mode == "forward-only" else nullcontext()
        with forward_context:
            logits = model.forward(x)
        if mode == "forward-only":
            if is_measurement:
                torch.cuda.synchronize()
                end = timeit.default_timer()
                specs["step"] = t - cfg.benchmark.warmup_steps
                specs["elapsed_seconds"] = end-start
                res.append(specs)
            continue

        loss = cross_entropy(logits, y)
        loss.backward()
        if mode == "forward-and-backward":
            if is_measurement:
                torch.cuda.synchronize()
                end = timeit.default_timer()
                specs["step"] = t - cfg.benchmark.warmup_steps
                specs["elapsed_seconds"] = end-start
                res.append(specs)
            continue

        optimizer.step()
        if mode == "full":
            if is_measurement:
                torch.cuda.synchronize()
                end = timeit.default_timer()
                specs["step"] = t - cfg.benchmark.warmup_steps
                specs["elapsed_seconds"] = end-start
                res.append(specs)
    res = pd.DataFrame(res)
    output_dir = HydraConfig.get().runtime.output_dir
    output_file_path = os.path.join(output_dir, f"{cfg.size}_{cfg.benchmark.mode}_warmup{cfg.benchmark.warmup_steps}.csv")
    res.to_csv(output_file_path, index=None)
    print(res)
    

if __name__ == "__main__":
    main()
