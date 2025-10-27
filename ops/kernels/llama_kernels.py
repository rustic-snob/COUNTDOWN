"""Kernel definitions for Forward/Backward kernels."""
from typing import Tuple

import torch
import triton
import triton.language as tl

# 1) Define “reasonable” tile‐sets per model (we avoid anything like 512×2048)


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

_model_tiles = {
    "llama3.1-8b": [(128, 64),
                    (64, 64),
                    (128, 8),
                    (256, 16),
                    (256, 32),
                    (256, 64),
                    (512, 16),
                    (512, 32), 
                    (512, 64),
                    (512, 128),
                    (16, 128),
                    (32, 128),
                    (64, 128),
                    (128, 128),
                    (256, 128),
                    (1024, 128),
                    (1024, 64),
                    (1024, 32),
                    (1024, 16)],
    "gemma-2-9b":  [(128, 64),
                    (64, 64),
                    (128, 8),
                    (256, 16),
                    (256, 32),
                    (256, 64),
                    (512, 16),
                    (512, 32), 
                    (512, 64),
                    (512, 128),
                    (16, 128),
                    (32, 128),
                    (64, 128),
                    (128, 128),
                    (256, 128)],
    "Qwen2.5-14b":[(128, 64),
                    (64, 64),
                    (128, 8),
                    (256, 16),
                    (256, 32),
                    (256, 64),
                    (512, 16),
                    (512, 32), 
                    (512, 64),
                    (512, 128),
                    (16, 128),
                    (32, 128),
                    (64, 128),
                    (128, 128),
                    (256, 128)],
    "phi-4":      [(128, 64),
                    (64, 64),
                    (128, 8),
                    (256, 16),
                    (256, 32),
                    (256, 64),
                    (512, 16),
                    (512, 32), 
                    (512, 64),
                    (512, 128),
                    (16, 128),
                    (32, 128),
                    (64, 128),
                    (128, 128),
                    (256, 128)],
}

# 2) Warp & stage sweep
_warps  = [2, 4, 8]

# 3) Build a full configs dict for each model
_configs_map = {}
_configs_map_pre_hook = {}
for model, tiles in _model_tiles.items():
    cfgs = []
    cfgs_pre_hook = []
    for (bn, bm) in tiles:
        for num_warps in _warps:
            cfgs.append(
                triton.Config(
                    {"BLOCK_SIZE_N": bn,
                        "BLOCK_SIZE_M": bm},
                    num_warps= num_warps,)
            )
            cfgs_pre_hook.append(
                triton.Config(
                    {"BLOCK_SIZE_N": bn,
                        "BLOCK_SIZE_M": bm},
                    num_warps=num_warps,
                    pre_hook=init_to_zero("Y"),
                )
            )
    _configs_map[model] = cfgs
    _configs_map_pre_hook[model] = cfgs_pre_hook

# 4) At import time, pick based on your MODEL_NAME env var
MODEL = "llama3.1-8b"  # or "gemma-2-9b" or "Qwen2.5-14b" or "phi-4"
configs = _configs_map.get(MODEL, _configs_map["llama3.1-8b"])
configs_pre_hook = _configs_map_pre_hook.get(MODEL, _configs_map["llama3.1-8b"])

@triton.jit
def silu(x):
    return (x * tl.sigmoid(x))


@triton.jit
def triton_silu_grad(x):
    return (tl.sigmoid(x) + (x * tl.sigmoid(x) * (1.0 - tl.sigmoid(x))))

# fmt: off
@triton.autotune(
    configs=configs,
    key=['CACHE_KEY_N', 'CACHE_KEY_M'],
)
# @triton.heuristics(
#     {
#         "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
#     }
# )
@triton.jit
def kernel_llama_dcidx_up(
    x_ptr, w_gate_ptr, w_up_ptr, out_ptr, sparse_idx_ptr,
    w_gate_stride_k,
    w_up_stride_k,
    N, M,
    CACHE_KEY_N,
    CACHE_KEY_M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    # EVEN_N: tl.constexpr,
):
    pid = tl.program_id(0)
    m_start = pid * BLOCK_SIZE_M
    offsets_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    
    sparse_idx_ptrs = sparse_idx_ptr + offsets_m
    sparse_idx = tl.load(sparse_idx_ptrs, mask=offsets_m < M, other=0)

    # Initialize accumulators
    gate = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    up = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    x_ptrs = x_ptr + offsets_n
    w_gate_ptrs = w_gate_ptr + offsets_n[None, :] + sparse_idx[:, None] * w_gate_stride_k
    w_up_ptrs = w_up_ptr + offsets_n[None, :] + sparse_idx[:, None] * w_up_stride_k

    # Process N dimension in tiles
    for n_start in range(N, 0, -BLOCK_SIZE_N):
        # Load x
        x = tl.load(x_ptrs) #if EVEN_N else tl.load(x_ptrs, mask=offsets_n < n_start, other=0.0)  # Shape: [BLOCK_SIZE_N]
        w_gate = tl.load(w_gate_ptrs) #if EVEN_N else tl.load(w_gate_ptrs, mask=offsets_n < n_start, other=0.0)
        w_up = tl.load(w_up_ptrs) #if EVEN_N else tl.load(w_up_ptrs, mask=offsets_n < n_start, other=0.0)

        # Compute partial sums
        gate += tl.sum(w_gate * x[None, :], axis=1)
        up += tl.sum(w_up * x[None, :], axis=1)
        
        x_ptrs += BLOCK_SIZE_N
        w_gate_ptrs += BLOCK_SIZE_N
        w_up_ptrs += BLOCK_SIZE_N

    # Apply activation function
    up *= silu(gate)

    # Write results
    tl.store(out_ptr + offsets_m, up, mask=offsets_m<M)

@torch.compile(fullgraph=True)
def llama_dcidx_up(x, w_gate_t, w_up_t, sparse_idx):
    K, N = w_up_t.shape  # w_up shape: [K, N]
    M = sparse_idx.shape[0]
    out = torch.empty((M), device=x.device, dtype=torch.float32)

    x = x.squeeze().contiguous()

    # Update strides after transposition (K * N)
    w_gate_stride_k = w_gate_t.stride(0)
    w_up_stride_k = w_up_t.stride(0)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

    kernel_llama_dcidx_up[grid](
        x, w_gate_t, w_up_t, out, sparse_idx,
        w_gate_stride_k,
        w_up_stride_k,
        N, M,
        N // 1024, M // 512,
    )

    out = out.view(1, 1, M)
    return out

@triton.autotune(
    configs=configs,
    key=['CACHE_KEY_N', 'CACHE_KEY_M'],
)
# @triton.heuristics(
#     {
#         "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
#     }
# )
@triton.jit
def kernel_llama_dcidx_down(
    x_ptr, w_down_ptr, out_ptr, sparse_idx_ptr,  # Pointers to matrices
    w_down_stride_k,
    # Matrix dimensions
    N,
    M,
    CACHE_KEY_N,
    CACHE_KEY_M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    # EVEN_N: tl.constexpr,
):

    """
    Kernel for computing Y = A[IDX, :]^T @ X + BIAS, where A is a dense matrix
    with Z rows and N columns. We also batch across the batch dimension of the input X.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, M)
    - Weight has shape (Z, N)
    - IDX has shape (M), where M is the number of non-zero rows in A
    - Bias has shape (N)
    - Output has shape (BATCHSIZE, N)
    """
    n_start = tl.program_id(0)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A
    offsets_m = tl.arange(0, BLOCK_SIZE_M)
    offsets_n = n_start * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    x_ptrs = x_ptr + offsets_m
    sparse_idx_ptrs = sparse_idx_ptr + offsets_m
    
    down = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for m_start in range(M, 0, -BLOCK_SIZE_M):
        idx = tl.load(sparse_idx_ptrs, mask=offsets_m < m_start, other=0)
        w_down_ptrs = w_down_ptr + (idx[:, None] * w_down_stride_k + offsets_n[None, :])
        x0 = tl.load(x_ptrs, mask=offsets_m < m_start, other=0.0)
        a = (
            tl.load(w_down_ptrs)
            # if EVEN_N
            # else tl.load(w_down_ptrs, mask=offsets_n[None, :] < N, other=0.0)
        )
        down += tl.sum(a * x0[:, None], 0)
        sparse_idx_ptrs += BLOCK_SIZE_M
        x_ptrs += BLOCK_SIZE_M

    tl.store(out_ptr + offsets_n, down, mask=offsets_n < N)

@torch.compile(fullgraph=True)
def llama_dcidx_down(x, w_down, sparse_idx):
    K, N = w_down.shape
    M = sparse_idx.shape[0]
    w_down_stride_k = w_down.stride(0) 

    kernel_type = "deterministic"
    # kernel_type = "atomicadd"  # This always seems to be faster for now

    output = torch.empty(
        N, device=x.device, dtype=torch.float32
    )

    # 1D launch kernel where each block gets its own program.
    if kernel_type == "deterministic":
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE_N"]),)  # noqa
    else:
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )  # noqa

    kernel = (
        kernel_llama_dcidx_down
        # if kernel_type == "deterministic"
        # else down_kernel_atomic_p_sparse_idx
    )
    kernel[grid](x, w_down, output, sparse_idx,
        w_down_stride_k,
        N, M,
        N // 32,
        M // 1024,  # key for triton cache (limit number of compilations)
    )
    output = output.view(1, 1, N)
    return output

@triton.autotune(
    configs=configs,
    key=['CACHE_KEY_N', 'CACHE_KEY_M'],
)
# @triton.heuristics(
#     {
#         "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
#     }
# )
@triton.jit
def kernel_llama_cats_up(
    Y,  # Pointers to matrices
    A,
    X,
    X_1,
    IDX,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    # EVEN_N: tl.constexpr,
):
    """
    Kernel for computing Y = A[IDX, :] @ X) * X_1, where A is a
    dense matrix with M rows and N columns.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, N)
    - Input X_1 has shape (BATCHSIZE, M)
    - A has shape (M, N)
    - IDX has shape (M), where M is the flag for non-zero rows in A
    - Output has shape (BATCHSIZE, M)
    """
    # EVEN_N is asserted to be true
    
    start_m = tl.program_id(0)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A and B
    rm = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0) > 0
    A = A + (rm[:, None] * stride_am + rn[None, :])
    X_1 = X_1 + rm
    X = X + rn
    
    if BATCHSIZE == 1:
        acc0 = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        x1_0 = tl.load(X_1, mask=idx, other=0.0)
        i_mask = idx[:, None]
        for n in range(N, 0, -BLOCK_SIZE_N):
            a = tl.load(A, mask=i_mask, other=0.0)
            x0 = tl.load(X)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
            A += BLOCK_SIZE_N
            X += BLOCK_SIZE_N
        acc_0 = acc0 * x1_0.to(tl.float32)
    elif BATCHSIZE == 2:
        acc0 = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        idx_1 = tl.load(IDX + M, mask=rm < M, other=0) > 0
        x1_0 = tl.load(X_1, mask=idx, other=0.0)
        x1_1 = tl.load(X_1 + M, mask=idx_1, other=0.0)
        i_mask = (idx | idx_1)[:, None]
        for n in range(N, 0, -BLOCK_SIZE_N):
            a = tl.load(A, mask=i_mask, other=0.0).to(tl.float32)
            x0_0 = tl.load(X)
            x0_1 = tl.load(X + N)
            acc0 += tl.sum(a * x0_0.to(tl.float32)[None, :], 1)
            acc1 += tl.sum(a * x0_1.to(tl.float32)[None, :], 1)
            A += BLOCK_SIZE_N
            X += BLOCK_SIZE_N
        acc_0 = acc0 * x1_0.to(tl.float32)
        acc_1 = acc1 * x1_1.to(tl.float32)

    elif BATCHSIZE == 3:
        acc0 = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        idx_1 = tl.load(IDX + M, mask=rm < M, other=0) > 0
        idx_2 = tl.load(IDX + 2 * M, mask=rm < M, other=0) > 0
        x1_0 = tl.load(X_1, mask=idx, other=0.0)
        x1_1 = tl.load(X_1 + M, mask=idx_1, other=0.0)
        x1_2 = tl.load(X_1 + 2 * M, mask=idx_2, other=0.0)
        i_mask = (idx | idx_1 | idx_2)[:, None]
        for n in range(N, 0, -BLOCK_SIZE_N):
            a = tl.load(A, mask=i_mask, other=0.0).to(tl.float32)
            x0_0 = tl.load(X)
            x0_1 = tl.load(X + N)
            x0_2 = tl.load(X + 2 * N)
            acc0 += tl.sum(a * x0_0.to(tl.float32)[None, :], 1)
            acc1 += tl.sum(a * x0_1.to(tl.float32)[None, :], 1)
            acc2 += tl.sum(a * x0_2.to(tl.float32)[None, :], 1)
            A += BLOCK_SIZE_N
            X += BLOCK_SIZE_N
        acc_0 = acc0 * x1_0.to(tl.float32)
        acc_1 = acc1 * x1_1.to(tl.float32)
        acc_2 = acc2 * x1_2.to(tl.float32)    
    elif BATCHSIZE == 4:
        acc0 = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        idx_1 = tl.load(IDX + M, mask=rm < M, other=0) > 0
        idx_2 = tl.load(IDX + 2 * M, mask=rm < M, other=0) > 0
        idx_3 = tl.load(IDX + 3 * M, mask=rm < M, other=0) > 0
        x1_0 = tl.load(X_1, mask=idx, other=0.0)
        x1_1 = tl.load(X_1 + M, mask=idx_1, other=0.0)
        x1_2 = tl.load(X_1 + 2 * M, mask=idx_2, other=0.0)
        x1_3 = tl.load(X_1 + 3 * M, mask=idx_3, other=0.0)
        i_mask = (idx | idx_1 | idx_2 | idx_3)[:, None]
        for n in range(N, 0, -BLOCK_SIZE_N):
            a = tl.load(A, mask=i_mask, other=0.0).to(tl.float32)
            x0_0 = tl.load(X)
            x0_1 = tl.load(X + N)
            x0_2 = tl.load(X + 2 * N)
            x0_3 = tl.load(X + 3 * N)
            acc0 += tl.sum(a * x0_0.to(tl.float32)[None, :], 1)
            acc1 += tl.sum(a * x0_1.to(tl.float32)[None, :], 1)
            acc2 += tl.sum(a * x0_2.to(tl.float32)[None, :], 1)
            acc3 += tl.sum(a * x0_3.to(tl.float32)[None, :], 1)
            A += BLOCK_SIZE_N
            X += BLOCK_SIZE_N
        acc_0 = acc0 * x1_0.to(tl.float32)
        acc_1 = acc1 * x1_1.to(tl.float32)
        acc_2 = acc2 * x1_2.to(tl.float32)
        acc_3 = acc3 * x1_3.to(tl.float32)

    # rematerialize rm and rn to save registers
    rm = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # write back result
    Y = Y + rm
    # acc = acc0 * x1
    tl.store(Y, acc_0, mask=rm < M)
    if BATCHSIZE >= 2:
        tl.store(Y + M, acc_1, mask=rm < M)
    if BATCHSIZE >= 3:
        tl.store(Y + 2 * M, acc_2, mask=rm < M)
    if BATCHSIZE >= 4:
        tl.store(Y + 3 * M, acc_3, mask=rm < M)

def llama_cats_up(
    x: torch.Tensor,
    x_1: torch.Tensor,
    wup: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute y = activation(x @ wgate[idx, :].T) * (x @ wup[idx, :].T).
    :param x: input tensor, (batch, N)
    :param x_1: input tensor, (batch, Z)
    :param wup: up weigth matrix, (Z, N)
    :param idx: flags, (Z,)
    :return: result tensor, (batch, N)
    """
    Z, N = wup.shape
    beam_width, seq_len, _ = x.shape
    # assert x.shape == (batch, N)
    # assert x_1.shape == (batch, Z)
    assert seq_len == 1
    assert beam_width >= 1 and beam_width <= 4
    x = x.contiguous()
    x_1 = x_1.contiguous()
    if wup.stride(1) > 1:
        wup = wup.contiguous()
    assert (
        x.dtype == wup.dtype
    ), f"Input and weight must have the same dtype, got {x.dtype} and {wup.dtype}"

    output = torch.empty(beam_width, seq_len, Z, device=x.device, dtype=torch.float32)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(Z, META["BLOCK_SIZE_M"]),)  # noqa

    kernel_llama_cats_up[grid](
        output,  # data ptrs
        wup,
        x,
        x_1,
        idx,
        Z,  # shapes
        N,
        Z // 512,  # key for triton cache (limit number of compilations)
        N // 1024,  # key for triton cache (limit number of compilations)
        wup.stride(0),  # strides
        beam_width,  # Can't use kwargs because auto-tuner requires args
    )
    return output.to(x.dtype)

@triton.autotune(
    configs=configs_pre_hook,
    key=['CACHE_KEY_N', 'CACHE_KEY_M'],
)
# @triton.heuristics(
#     {
#         "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
#     }
# )
@triton.jit
def kernel_llama_down(
    Y,  # Pointers to matrices
    A,
    X,
    IDX,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    # EVEN_N: tl.constexpr,
):

    """
    Kernel for computing Y = A[IDX, :]^T @ X + BIAS, where A is a dense matrix
    with Z rows and N columns. We also batch across the batch dimension of the input X.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, M)
    - Weight has shape (Z, N)
    - IDX has shape (M), where M is the number of non-zero rows in A
    - Bias has shape (N)
    - Output has shape (BATCHSIZE, N)
    """
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A
    rm = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = start_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0) > 0
    A = A + (rm[:, None] * stride_am + rn[None, :])
    X = X + rm
    Y = Y + rn
    
    a = tl.load(A, mask=idx[:, None], other=0.0)
    x0 = tl.load(X)#, mask=idx, other=0.0) # if flag_gemv is correct, this will be unnecessary.
    acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

    # rematerialize rm and rn to save registers
    rn = start_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.atomic_add(Y, acc0, mask=rn < N)

def llama_down(
    x: torch.Tensor,
    weight: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute y = weight[idx, :]^T @ x.
    :param x: input tensor
    :param weight: weight matrix
    :param idx: indices
    :return: result tensor
    """
    Z, N = weight.shape

    x = x.contiguous()
   
    output = torch.empty(
        1,
        1,
        N,
        device=x.device,
        dtype=torch.float32,
    )

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(Z, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )  # noqa

    kernel = kernel_llama_down
    kernel[grid](
        output,  # data ptrs
        weight,
        x,
        idx,
        Z,  # shapes
        N,
        Z // 128,  # key for triton cache (limit number of compilations)
        N // 32,
        weight.stride(0),  # strides
    )
    return output


@triton.autotune(
    configs=configs,
    key=['CACHE_KEY_N', 'CACHE_KEY_M'],
)
# @triton.heuristics(
#     {
#         "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
#     }
# )
@triton.jit
def kernel_llama_mc_up(
    Y,  # Pointers to matrices
    A,
    X,
    X_1,
    IDX,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    # EVEN_N: tl.constexpr,
):
    # EVEN_N is asserted to be true
    start_m = tl.program_id(0)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A and B
    rm = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0) > 0
    A = A + (rm[:, None] * stride_am + rn[None, :])
    X_1 = X_1 + rm
    X = X + rn
    
    acc0 = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    x1_0 = tl.load(X_1, mask=idx, other=0.0)
    i_mask = idx[:, None]
    for n in range(N, 0, -BLOCK_SIZE_N):
        a = tl.load(A, mask=i_mask, other=0.0)
        x0 = tl.load(X)
        acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
        A += BLOCK_SIZE_N
        X += BLOCK_SIZE_N
    acc_0 = silu(acc0) * x1_0.to(tl.float32)
    

    # rematerialize rm and rn to save registers
    rm = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # write back result
    Y = Y + rm
    # acc = acc0 * x1
    tl.store(Y, acc_0, mask=rm < M)

def llama_mc_up(
    x: torch.Tensor,
    x_1: torch.Tensor,
    wgate: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute y = activation(x @ wgate[idx, :].T) * (x @ wgate[idx, :].T).
    :param x: input tensor, (batch, N)
    :param x_1: input tensor, (batch, Z)
    :param wgate: up weigth matrix, (Z, N)
    :param idx: flags, (Z,)
    :return: result tensor, (batch, N)
    """
    Z, N = wgate.shape

    x = x.contiguous()
    x_1 = x_1.contiguous()

    output = torch.empty(1, 1, Z, device=x.device, dtype=torch.float32)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(Z, META["BLOCK_SIZE_M"]),)  # noqa

    kernel_llama_mc_up[grid](
        output,  # data ptrs
        wgate,
        x,
        x_1,
        idx,
        Z,  # shapes
        N,
        Z // 512,  # key for triton cache (limit number of compilations)
        N // 1024,  # key for triton cache (limit number of compilations)
        wgate.stride(0),  # strides
    )
    return output

@triton.autotune(
    configs=configs,
    key=['CACHE_KEY_N', 'CACHE_KEY_M'],
)
@triton.jit
def kernel_llama_dcmask_up(
    Y,  # Pointers to matrices
    WG,
    WU,
    X,
    IDX,
    # Matrix dimensions
    K,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_ak,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Kernel for computing Y = (WU[IDX, :] @ X) * (ACT(WU[IDX, :] @ X))
    - Input X has shape (BATCHSIZE, N)
    - W has shape (K, N)
    - IDX has shape (K), where M is the flag for non-zero rows in A
    - Output has shape (BATCHSIZE, K)
    """
    # EVEN_N is asserted to be true
    start_k = tl.program_id(0)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A and B
    rk = start_k * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)

    IDX = IDX + rk
    idx = tl.load(IDX, mask=rk < K, other=0) > 0
    WG = WG + (rk[:, None] * stride_ak + rn[None, :])
    WU = WU + (rk[:, None] * stride_ak + rn[None, :])
    X = X + rn
    acc_g = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    i_mask = idx[:, None]
    for n in range(N, 0, -BLOCK_SIZE_N):
        wg = tl.load(WG, mask=i_mask, other=0.0)
        wu = tl.load(WU, mask=i_mask, other=0.0)
        x0 = tl.load(X)
        acc_g += tl.sum(wg.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
        acc_u += tl.sum(wu.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
        WG += BLOCK_SIZE_N
        WU += BLOCK_SIZE_N
        X += BLOCK_SIZE_N

    acc = acc_u * silu(acc_g)

    Y = Y + rk
    
    tl.store(Y, acc, mask=rk < K)

def llama_dcmask_up(
    x: torch.Tensor,
    wgate: torch.Tensor,
    wup: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    K, N = wup.shape

    out = torch.empty(1, 1, K, device=x.device, dtype=torch.float32)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(K, META["BLOCK_SIZE_M"]),)  # noqa

    kernel_llama_dcmask_up[grid](
        out,  # data ptrs
        wgate,
        wup,
        x,
        idx,
        K,  # shapes
        N,
        K // 512,  # key for triton cache (limit number of compilations)
        N // 1024,  # key for triton cache (limit number of compilations)
        wup.stride(0),  # strides
    )
    return out

@triton.autotune(
    configs=configs_pre_hook,
    key=['CACHE_KEY_N', 'CACHE_KEY_M'],
)
@triton.jit
def kernel_llama_dcmask_down(
    Y,  # Pointers to matrices
    W,
    X,
    IDX,
    # Matrix dimensions
    K,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_ak,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):

    """
    Kernel for computing Y = A[IDX, :]^T @ X + BIAS, where A is a dense matrix
    with Z rows and N columns. We also batch across the batch dimension of the input X.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, K)
    - Weight has shape (K, N)
    - IDX has shape (K), where M is the flag for non-zero rows in A
    - Output has shape (BATCHSIZE, N)
    """
    start_k = tl.program_id(0)
    start_n = tl.program_id(1)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A
    rk = start_k * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = start_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    IDX = IDX + rk
    idx = tl.load(IDX, mask=rk < K, other=0) > 0
    W = W + (rk[:, None] * stride_ak + rn[None, :])
    X = X + rk
    Y = Y + rn
    
    a = tl.load(W, mask=idx[:, None], other=0.0)
    x0 = tl.load(X)
    acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

    # rematerialize rm and rn to save registers
    rn = start_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.atomic_add(Y, acc0, mask=rn < N)

def llama_dcmask_down(
    x: torch.Tensor,
    weight: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute y = weight[idx, :]^T @ x.
    :param x: input tensor
    :param weight: weight matrix
    :param idx: indices
    :return: result tensor
    """
    K, N = weight.shape

    out = torch.empty(
        1,
        1,
        N,
        device=x.device,
        dtype=torch.float32,
    )

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(K, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )  # noqa

    kernel = kernel_llama_dcmask_down
    kernel[grid](
        out,  # data ptrs
        weight,
        x,
        idx,
        K,  # shapes
        N,
        K // 128,  # key for triton cache (limit number of compilations)
        N // 32,
        weight.stride(0),  # strides
    )
    return out