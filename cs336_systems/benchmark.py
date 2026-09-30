import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW
from einops import einsum
from jaxtyping import Bool, Float
import hydra
import random
import numpy as np
import torch
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import timeit
import pandas as pd
import os
import torch.cuda.nvtx as nvtx
import math

def run_forward_only(model: torch.nn.Module, x: torch.Tensor) -> float:
    start = timeit.default_timer()
    with torch.no_grad():
        _ = model.forward(x)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        return end-start

def run_forward_and_backward(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    start = timeit.default_timer()
    logits = model.forward(x)
    loss = cross_entropy(logits, y)
    loss.backward()
    torch.cuda.synchronize()
    end = timeit.default_timer()
    return end-start

def run_full(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.optimizer) -> float:
    start = timeit.default_timer()
    logits = model.forward(x)
    loss = cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    end = timeit.default_timer()
    return end-start

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys    d_k"],
    V: Float[torch.Tensor, " ... keys    d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        res = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return res

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

    for _ in range(cfg.benchmark.warmup_steps):
        optimizer.zero_grad()
        if mode == "forward-only":
            run_forward_only(model, x)
        elif mode == "forward-and-backward":
            run_forward_and_backward(model, x, y)
        elif mode == "full":
            run_full(model, x, y)

    res = []
    for t in range(cfg.benchmark.measure_steps):
        specs = {
            "size": cfg.size,
            "d_model": cfg.model.d_model,
            "d_ff": cfg.model.d_ff,
            "num_layers": cfg.model.num_layers,
            "num_heads": cfg.model.num_heads,
            "context_length": cfg.model.context_length,
            "batch_size": cfg.training.batch_size,
            "mode": cfg.benchmark.mode
        }
        duration = 0.0
        optimizer.zero_grad()
        torch.cuda.synchronize()
        if mode == "forward-only":
            cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
            with nvtx.range("forward-only"):
                duration = run_forward_only(model, x)
        elif mode == "forward-and-backward":
            with nvtx.range("forward-and-backward"):
                duration = run_forward_and_backward(model, x, y)
        elif mode == "full":
            with nvtx.range("full"):
                duration = run_full(model, x, y)
        specs["step"] = t
        specs["elapsed_seconds"] = duration
        res.append(specs)
    res = pd.DataFrame(res)
    output_dir = HydraConfig.get().runtime.output_dir
    output_file_path = os.path.join(output_dir, f"{cfg.size}_{cfg.benchmark.mode}_warmup{cfg.benchmark.warmup_steps}.csv")
    res.to_csv(output_file_path, index=None)
    print(res)
    

if __name__ == "__main__":
    main()
