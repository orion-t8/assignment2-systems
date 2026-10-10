import torch
from torch import Tensor
from jaxtyping import Float
from einops import einsum
import math

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, Q: Float[Tensor, "batch_size seq_len d_k"],
                K: Float[Tensor, "batch_size seq_len d_k"],
                V: Float[Tensor, "batch_size seq_len d_v"], is_causal=False):
        B_q = B_k = 16 # tile size for Q and K resp.
        batch_size, n_queries, d_k = Q.shape
        _, n_keys, _ = K.shape
        input_dtype = Q.dtype
        output = torch.empty_like(V, dtype=input_dtype, device=V.device)
        L = torch.empty((batch_size, n_queries), dtype=torch.float32, device=V.device)
        for i in range(n_queries // B_q):
            Q_i = Q[:, (i*B_q):((i+1)*B_q), :]
            curr_O = torch.zeros((batch_size, B_q, V.shape[-1]), dtype=torch.float32, device=V.device) # running unnormalized output
            curr_l = torch.zeros((batch_size, B_q), dtype=torch.float32, device=V.device) # running sumexp
            curr_m = -torch.inf * torch.ones((batch_size, B_q), dtype=torch.float32, device=V.device) # running max
            for j in range(n_keys // B_k):
                K_j = K[:, (j*B_k):((j+1)*B_k), :]
                V_j = V[:, (j*B_k):((j+1)*B_k), :]
                S = einsum(Q_i, K_j, "batch B_q d_k, batch B_k d_k -> batch B_q B_k") / math.sqrt(d_k)
                m = torch.maximum(curr_m, torch.max(S, dim=-1)[0]) # (batch, B_q)
                P = torch.exp(S - m[..., None]) # (batch, B_q, B_k)
                l = torch.exp(curr_m - m) * curr_l + torch.sum(P, dim=-1) # (batch, B_q)
                O = torch.exp(curr_m - m)[..., None] * curr_O + einsum(P.to(input_dtype), V_j, "batch B_q B_k, batch B_k d_v -> batch B_q d_v")
                curr_m = m
                curr_l = l
                curr_O = O
            output[:, (i*B_q):((i+1)*B_q), :] = curr_O / curr_l[..., None]
            L[:, (i*B_q):((i+1)*B_q)] = curr_m + torch.log(curr_l)
        ctx.save_for_backward(L, Q, K, V, output)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError