import torch
from torch import Tensor
from jaxtyping import Float
from einops import einsum
import math
import triton
import triton.language as tl

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

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                Q: Float[Tensor, "batch N_QUERIES D"],
                K: Float[Tensor, "batch N_KEYS D"],
                V: Float[Tensor, "batch N_KEYS D"],
                is_causal=False):
        assert len(Q.shape) == len(K.shape) == len(V.shape) == 3
        assert Q.shape[0] == K.shape[0] == V.shape[0]
        assert K.shape[1] == V.shape[1]
        assert Q.shape[2] == K.shape[2] == V.shape[2]
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

        batch, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[1]
        O = torch.empty_like(Q, device=Q.device)
        L = torch.empty((batch, N_QUERIES), device=Q.device, dtype=torch.float32)

        ctx.Q_TILE_SIZE = 16        
        ctx.K_TILE_SIZE = 16
        ctx.is_causal = is_causal
        flash_fwd_kernel[(triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE), batch)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            1/math.sqrt(D),
            D,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            is_causal
        )
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    Ktrans_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    dtype = Q_block_ptr.type.element_ty
    Q = tl.load(Q_block_ptr)
    curr_O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    curr_l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    curr_m = tl.full((Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32)

    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Ktrans = tl.load(Ktrans_block_ptr)
        V = tl.load(V_block_ptr)
        S = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        S = tl.dot(Q, Ktrans, acc=S) * scale
        if is_causal:
            query_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            key_indices = tl.arange(0, K_TILE_SIZE) + i * K_TILE_SIZE
            S -=  (query_indices[:, None] < key_indices[None, :]) * 1e6
        m = tl.maximum(curr_m, tl.max(S, axis=1))
        P = tl.exp(S - m[:, None])
        l = tl.exp(curr_m - m) * curr_l + tl.sum(P, axis=1)
        curr_O *= tl.exp(curr_m - m)[:, None]
        curr_O = tl.dot(P.to(dtype=dtype), V, acc=curr_O)
        curr_m = m
        curr_l = l
        Ktrans_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr.advance((K_TILE_SIZE, 0))
    output = curr_O / curr_l[:, None]
    tl.store(O_block_ptr, output.to(dtype=dtype))
    tl.store(L_block_ptr, curr_m + tl.log(curr_l))