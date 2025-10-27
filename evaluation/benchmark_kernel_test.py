import os

import torch
import torch.nn as nn

import triton
import triton.language as tl

import pandas as pd

import gc

from tqdm.auto import tqdm

from ops.kernels.llama_kernels import llama_dcidx_up 
from ops.kernels.llama_kernels import llama_dcidx_down 
from ops.kernels.llama_kernels import llama_cats_up 
from ops.kernels.llama_kernels import llama_down 
from ops.kernels.llama_kernels import llama_mc_up 
from ops.kernels.llama_kernels import llama_dcmask_up 
from ops.kernels.llama_kernels import llama_dcmask_down

from ops.kernels.gemma_kernels import gemma_dcidx_up
from ops.kernels.gemma_kernels import gemma_dcidx_down
from ops.kernels.gemma_kernels import gemma_cats_up
from ops.kernels.gemma_kernels import gemma_down
from ops.kernels.gemma_kernels import gemma_mc_up
from ops.kernels.gemma_kernels import gemma_dcmask_up
from ops.kernels.gemma_kernels import gemma_dcmask_down

from ops.kernels.qwen_kernels import qwen_dcidx_up
from ops.kernels.qwen_kernels import qwen_dcidx_down
from ops.kernels.qwen_kernels import qwen_cats_up
from ops.kernels.qwen_kernels import qwen_down
from ops.kernels.qwen_kernels import qwen_mc_up
from ops.kernels.qwen_kernels import qwen_dcmask_up
from ops.kernels.qwen_kernels import qwen_dcmask_down

from ops.kernels.phi_kernels import phi_dcidx_up
from ops.kernels.phi_kernels import phi_dcidx_down
from ops.kernels.phi_kernels import phi_cats_up
from ops.kernels.phi_kernels import phi_down
from ops.kernels.phi_kernels import phi_mc_up
from ops.kernels.phi_kernels import phi_dcmask_up
from ops.kernels.phi_kernels import phi_dcmask_down



@triton.jit
def silu(x):
    return (x * tl.sigmoid(x))

def llama_fused_triton_sparsed(x, w_gate, w_up, w_down, sparse_index):
    return llama_dcidx_down(llama_dcidx_up(x, w_gate,
                                            w_up,
                                            sparse_index),
                                        w_down, sparse_index)

def llama_gemv_gemv_triton(x, x_1, Wup, Wdownt, threshold):
    flags = torch.abs(x_1) > threshold
    x = llama_cats_up(x, x_1, Wup, flags)
    return llama_down(x, Wdownt, flags)

def llama_m_countdown_gemv_gemv_triton(x, x_1, Wgate, Wdownt, threshold):
    flags = torch.abs(x_1) > threshold
    x = llama_mc_up(x, x_1, Wgate, flags)
    return llama_down(x, Wdownt, flags)

def llama_method_sparse_prediction_unified(x, predictor_u, predictor_v):
    x = x.squeeze()
    prac = (x @ predictor_u @ predictor_v)
    column_used = (prac > 0.5)
    return torch.nonzero(column_used)

def llama_method_sparse_masking_unified(x, predictor_u, predictor_v, threshold):
    prac = (x @ predictor_u @ predictor_v)
    flags = (prac > threshold)
    return flags

def llama_method_cats_triton(x, Wgate, Wup, Wdownt, threshold, act_fn):
    x_1 = act_fn(torch.matmul(x, Wgate))
    return llama_gemv_gemv_triton(x, x_1, Wup, Wdownt, threshold)

def llama_method_m_countdown_triton(x, Wup, Wgate, Wdown, threshold, act_fn):
    x_1 = torch.matmul(x, Wup)
    return llama_m_countdown_gemv_gemv_triton(x, x_1, Wgate, Wdown, threshold)

def llama_method_masking_fused_dcountdown_triton(x, Wgate, Wup, Wdown, predictor_u, predictor_v, threshold):
    flags = llama_method_sparse_masking_unified(x, predictor_u, predictor_v, threshold)
    x = llama_dcmask_up(x, Wgate, Wup, flags)
    return llama_dcmask_down(x, Wdown, flags)

def gemma_fused_triton_sparsed(x, w_gate, w_up, w_down, sparse_index):
    return gemma_dcidx_down(gemma_dcidx_up(x, w_gate,
                                            w_up,
                                            sparse_index),
                                        w_down, sparse_index)

def gemma_gemv_gemv_triton(x, x_1, Wup, Wdownt, threshold):
    flags = torch.abs(x_1) > threshold
    x = gemma_cats_up(x, x_1, Wup, flags)
    return gemma_down(x, Wdownt, flags)

def gemma_m_countdown_gemv_gemv_triton(x, x_1, Wgate, Wdownt, threshold):
    flags = torch.abs(x_1) > threshold
    x = gemma_mc_up(x, x_1, Wgate, flags)
    return gemma_down(x, Wdownt, flags)

def gemma_method_sparse_prediction_unified(x, predictor_u, predictor_v):
    x = x.squeeze()
    prac = (x @ predictor_u @ predictor_v)
    column_used = (prac > 0.5)
    return torch.nonzero(column_used)

def gemma_method_sparse_masking_unified(x, predictor_u, predictor_v, threshold):
    prac = (x @ predictor_u @ predictor_v)
    flags = (prac > threshold)
    return flags

def gemma_method_cats_triton(x, Wgate, Wup, Wdownt, threshold, act_fn):
    x_1 = act_fn(torch.matmul(x, Wgate))
    return gemma_gemv_gemv_triton(x, x_1, Wup, Wdownt, threshold)

def gemma_method_m_countdown_triton(x, Wup, Wgate, Wdown, threshold, act_fn):
    x_1 = torch.matmul(x, Wup)
    return gemma_m_countdown_gemv_gemv_triton(x, x_1, Wgate, Wdown, threshold)

def gemma_method_masking_fused_dcountdown_triton(x, Wgate, Wup, Wdown, predictor_u, predictor_v, threshold):
    flags = gemma_method_sparse_masking_unified(x, predictor_u, predictor_v, threshold)
    x = gemma_dcmask_up(x, Wgate, Wup, flags)
    return gemma_dcmask_down(x, Wdown, flags)

def qwen_fused_triton_sparsed(x, w_gate, w_up, w_down, sparse_index):
    return qwen_dcidx_down(qwen_dcidx_up(x, w_gate,
                                            w_up,
                                            sparse_index),
                                        w_down, sparse_index)

def qwen_gemv_gemv_triton(x, x_1, Wup, Wdownt, threshold):
    flags = torch.abs(x_1) > threshold
    x = qwen_cats_up(x, x_1, Wup, flags)
    return qwen_down(x, Wdownt, flags)

def qwen_m_countdown_gemv_gemv_triton(x, x_1, Wgate, Wdownt, threshold):
    flags = torch.abs(x_1) > threshold
    x = qwen_mc_up(x, x_1, Wgate, flags)
    return qwen_down(x, Wdownt, flags)

def qwen_method_sparse_prediction_unified(x, predictor_u, predictor_v):
    x = x.squeeze()
    prac = (x @ predictor_u @ predictor_v)
    column_used = (prac > 0.5)
    return torch.nonzero(column_used)

def qwen_method_sparse_masking_unified(x, predictor_u, predictor_v, threshold):
    prac = (x @ predictor_u @ predictor_v)
    flags = (prac > threshold)
    return flags

def qwen_method_cats_triton(x, Wgate, Wup, Wdownt, threshold, act_fn):
    x_1 = act_fn(torch.matmul(x, Wgate))
    return qwen_gemv_gemv_triton(x, x_1, Wup, Wdownt, threshold)

def qwen_method_m_countdown_triton(x, Wup, Wgate, Wdown, threshold, act_fn):
    x_1 = torch.matmul(x, Wup)
    return qwen_m_countdown_gemv_gemv_triton(x, x_1, Wgate, Wdown, threshold)

def qwen_method_masking_fused_dcountdown_triton(x, Wgate, Wup, Wdown, predictor_u, predictor_v, threshold):
    flags = qwen_method_sparse_masking_unified(x, predictor_u, predictor_v, threshold)
    x = qwen_dcmask_up(x, Wgate, Wup, flags)
    return qwen_dcmask_down(x, Wdown, flags)

def phi_fused_triton_sparsed(x, w_gate, w_up, w_down, sparse_index):
    return phi_dcidx_down(phi_dcidx_up(x, w_gate,
                                            w_up,
                                            sparse_index),
                                        w_down, sparse_index)

def phi_gemv_gemv_triton(x, x_1, Wup, Wdownt, threshold):
    flags = torch.abs(x_1) > threshold
    x = phi_cats_up(x, x_1, Wup, flags)
    return phi_down(x, Wdownt, flags)

def phi_m_countdown_gemv_gemv_triton(x, x_1, Wgate, Wdownt, threshold):
    flags = torch.abs(x_1) > threshold
    x = phi_mc_up(x, x_1, Wgate, flags)
    return phi_down(x, Wdownt, flags)

def phi_method_sparse_prediction_unified(x, predictor_u, predictor_v):
    x = x.squeeze()
    prac = (x @ predictor_u @ predictor_v)
    column_used = (prac > 0.5)
    return torch.nonzero(column_used)

def phi_method_sparse_masking_unified(x, predictor_u, predictor_v, threshold):
    prac = (x @ predictor_u @ predictor_v)
    flags = (prac > threshold)
    return flags

def phi_method_cats_triton(x, Wgate, Wup, Wdownt, threshold, act_fn):
    x_1 = act_fn(torch.matmul(x, Wgate))
    return phi_gemv_gemv_triton(x, x_1, Wup, Wdownt, threshold)

def phi_method_m_countdown_triton(x, Wup, Wgate, Wdown, threshold, act_fn):
    x_1 = torch.matmul(x, Wup)
    return phi_m_countdown_gemv_gemv_triton(x, x_1, Wgate, Wdown, threshold)

def phi_method_masking_fused_dcountdown_triton(x, Wgate, Wup, Wdown, predictor_u, predictor_v, threshold):
    flags = phi_method_sparse_masking_unified(x, predictor_u, predictor_v, threshold)
    x = phi_dcmask_up(x, Wgate, Wup, flags)
    return phi_dcmask_down(x, Wdown, flags)

def method_pytorch_full(x, up_proj, gate_proj, act, down_proj):
    return down_proj(act(gate_proj(x)) * up_proj(x))

def method_pytorch_full_mat(x, w_gate, w_up, w_down):
    return torch.matmul((act(torch.matmul(x, w_gate)) * torch.matmul(x, w_up)), w_down)


# Benchmark function
def benchmark(func, *args, num_runs=500):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_runs):
        result = func(*args)
        torch.cuda.synchronize()
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_runs  # Average time per run in milliseconds

model_config = {
    'llama3.1-8b': (4096, 14336, nn.SiLU()),
    'gemma-2-9b': (3584, 14336, nn.GELU(approximate='tanh')),
    'Qwen2.5-14b': (5120, 13824, nn.SiLU()),
    'phi-4': (5120, 17920, nn.SiLU()),
}

df = pd.DataFrame(columns=['model_name', 'sparsity', 'Cats', 'M-Countdown', 'D-Countdown', 'Masking D-Countdown', 'PyTorch Opt Full', 'PyTorch Opt Full Mat', 'Pytorch Full', 'Pytorch Full Mat'])
r = 512

select_kernels = {"llama3.1-8b": (llama_method_sparse_prediction_unified, llama_method_cats_triton, llama_method_m_countdown_triton, llama_method_masking_fused_dcountdown_triton, llama_fused_triton_sparsed),
                    "gemma-2-9b": (gemma_method_sparse_prediction_unified, gemma_method_cats_triton, gemma_method_m_countdown_triton, gemma_method_masking_fused_dcountdown_triton, gemma_fused_triton_sparsed),
                    "Qwen2.5-14b": (qwen_method_sparse_prediction_unified, qwen_method_cats_triton, qwen_method_m_countdown_triton, qwen_method_masking_fused_dcountdown_triton, qwen_fused_triton_sparsed),
                    "phi-4": (phi_method_sparse_prediction_unified, phi_method_cats_triton, phi_method_m_countdown_triton, phi_method_masking_fused_dcountdown_triton, phi_fused_triton_sparsed)}

for model_name, (n, d, act) in model_config.items():
    x = torch.randn(1, 1, n, device='cuda')
    up_proj = nn.Linear(n, d, bias=False, device='cuda')
    gate_proj = nn.Linear(n, d, bias=False, device='cuda')
    down_proj = nn.Linear(d, n, bias=False, device='cuda')
    up_proj.weight = nn.Parameter(torch.randn(d, n, device='cuda'))
    gate_proj.weight = nn.Parameter(torch.randn(d, n, device='cuda'))
    down_proj.weight = nn.Parameter(torch.randn(n, d, device='cuda'))

    with torch.no_grad():
        for _ in range(50):
            method_pytorch_full(x, up_proj, gate_proj, act, down_proj)
            method_pytorch_full_mat(x.squeeze(1), up_proj.weight.t().contiguous(), gate_proj.weight.t().contiguous(), down_proj.weight.t().contiguous())

        # Benchmark
        time_full_pytorch = benchmark(method_pytorch_full, x, up_proj, gate_proj, act, down_proj)
        time_full_pytorch_mat = benchmark(method_pytorch_full_mat, x.squeeze(1), up_proj.weight.t().contiguous(), gate_proj.weight.t().contiguous(), down_proj.weight.t().contiguous())

    del x
    torch.cuda.empty_cache()
    gc.collect()

    method_sparse_prediction_unified, method_cats_triton, method_m_countdown_triton, method_masking_fused_dcountdown_triton, fused_triton_sparsed = select_kernels[model_name]

    for SPARSITY in tqdm([i/10 for i in range(1, 10)], desc=f"Testing Sparsity of {model_name}"):
        num_used_columns = int(d * (1 - SPARSITY))

        x = torch.randn(1, 1, n, device='cuda')
        up_small_proj = nn.Linear(n, num_used_columns, bias=False, device='cuda')
        gate_small_proj = nn.Linear(n, num_used_columns, bias=False, device='cuda')
        down_small_proj = nn.Linear(num_used_columns, n, bias=False, device='cuda')
        up_small_proj.weight = nn.Parameter(torch.randn(num_used_columns, n, device='cuda'))
        gate_small_proj.weight = nn.Parameter(torch.randn(num_used_columns, n, device='cuda'))
        down_small_proj.weight = nn.Parameter(torch.randn(n, num_used_columns, device='cuda'))

        column_used = torch.randperm(d)[:num_used_columns].cuda()

        predictor_u = torch.randn(n, r, device='cuda')
        predictor_v = torch.randn(r, d, device='cuda')

        # Threshold should be the value of num_used_columns th largest element
        cats_threshold = torch.topk(torch.abs(act(x.squeeze() @ gate_proj.weight.t().contiguous())), num_used_columns).values[-1].item()
        m_countdown_threshold = torch.topk(torch.abs(x.squeeze() @ up_proj.weight.t().contiguous()), num_used_columns).values[-1].item()
        d_countdown_threshold = torch.topk((x.squeeze() @ predictor_u @ predictor_v), num_used_columns).values[-1].item()

        with torch.no_grad():
            for _ in range(50):
                method_sparse_prediction_unified(x, predictor_u, predictor_v)
                fused_triton_sparsed(x, gate_proj.weight, up_proj.weight, down_proj.weight.t().contiguous(), column_used)
                method_pytorch_full(x, up_small_proj, gate_small_proj, act, down_small_proj)
                method_pytorch_full_mat(x.squeeze(1), up_small_proj.weight.t().contiguous(), gate_small_proj.weight.t().contiguous(), down_small_proj.weight.t().contiguous())
                method_cats_triton(x, gate_proj.weight.t().contiguous(), up_proj.weight, down_proj.weight.t().contiguous(), cats_threshold, act)
                method_m_countdown_triton(x, up_proj.weight.t().contiguous(), gate_proj.weight, down_proj.weight.t().contiguous(), m_countdown_threshold, act)
                method_masking_fused_dcountdown_triton(x, gate_proj.weight, up_proj.weight, down_proj.weight.t().contiguous(), predictor_u, predictor_v, d_countdown_threshold)
            # Benchmark
            time_sparse_unified = benchmark(method_sparse_prediction_unified, x, predictor_u, predictor_v)
            time_fused_triton = benchmark(fused_triton_sparsed, x, gate_proj.weight, up_proj.weight, down_proj.weight.t().contiguous(), column_used)
            time_full_opt_pytorch = benchmark(method_pytorch_full, x, up_small_proj, gate_small_proj, act, down_small_proj)
            time_full_opt_pytorch_mat = benchmark(method_pytorch_full_mat, x.squeeze(1), up_small_proj.weight.t().contiguous(), gate_small_proj.weight.t().contiguous(), down_small_proj.weight.t().contiguous())
            time_cats_triton = benchmark(method_cats_triton, x, gate_proj.weight.t().contiguous(), up_proj.weight, down_proj.weight.t().contiguous(), cats_threshold, act)
            time_m_countdown_triton = benchmark(method_m_countdown_triton, x, up_proj.weight.t().contiguous(), gate_proj.weight, down_proj.weight.t().contiguous(), m_countdown_threshold, act)
            time_masking_d_countdown = benchmark(method_masking_fused_dcountdown_triton, x, gate_proj.weight, up_proj.weight, down_proj.weight.t().contiguous(), predictor_u, predictor_v, d_countdown_threshold)
            
            time_d_countdown = time_fused_triton+time_sparse_unified

        data = {
            'model_name': model_name,
            'sparsity': SPARSITY,
            'Cats': time_cats_triton,
            'M-Countdown': time_m_countdown_triton,
            'D-Countdown': time_d_countdown,
            'Masking D-Countdown': time_masking_d_countdown,
            'PyTorch Opt Full': time_full_opt_pytorch,
            'PyTorch Opt Full Mat': time_full_opt_pytorch_mat,
            'Pytorch Full': time_full_pytorch,
            'Pytorch Full Mat': time_full_pytorch_mat
        }

        df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)

        del x, up_small_proj, gate_small_proj, down_small_proj, column_used, predictor_u, predictor_v, cats_threshold, m_countdown_threshold, d_countdown_threshold
        torch.cuda.empty_cache()
        gc.collect()

    del up_proj, gate_proj, down_proj
    torch.cuda.empty_cache()
    gc.collect()

df.to_csv('kernel_benchmark_results.csv', index=False)