import gc
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
from typing import Optional, Tuple, Union, List
from transformers.cache_utils import Cache, DynamicCache

def _decompose_weight(weight, r):
    torch.nn.init.xavier_uniform_(weight)
    dtype = weight.dtype
    weight_full = weight.detach().to(torch.float32)
    u_full, s_full, v_full = torch.pca_lowrank(weight_full, q=r)
    u_full = u_full * torch.sqrt(s_full).unsqueeze(0)
    v_full = v_full * torch.sqrt(s_full).unsqueeze(0)
    u = u_full.to(dtype)
    v = v_full.to(dtype)
    u = u.to('cuda')
    v = v.to('cuda')
    u.requires_grad = True
    v.requires_grad = True    
    del u_full, s_full, v_full, weight_full
    
    torch.cuda.empty_cache()
    gc.collect()
        
    return (u, v)

def _activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def _weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        # x_quant = x + (_activation_quant(x) - x).detach()
        w_quant = w + (_weight_quant(w) - w).detach()
        # y = F.linear(x_quant, w_quant)
        y = F.linear(x, w_quant)
        return y

def _make_bitlinear(weight):
    bitlinear = BitLinear(weight.shape[1], weight.shape[0]).to(weight.device).to(dtype=weight.dtype)
    bitlinear.weight = weight
    bitlinear.bias = None
    bitlinear.weight.requires_grad = True

    return bitlinear

def init_predictors(model, predictor_shape, r):
    if predictor_shape == 'lowrank':
        return [_decompose_weight(layer.mlp.up_proj.weight, r) for layer in tqdm(model.model.layers, desc="Awakening predictors...")]
    elif predictor_shape == 'full':
        return [torch.nn.init.xavier_uniform_(layer.mlp.up_proj.weight) for layer in tqdm(model.model.layers, desc="Awakening predictors...")]
    elif predictor_shape == 'kan':
        return [KAN([layer.mlp.up_proj.weight.shape[1], r, layer.mlp.up_proj.weight.shape[0]], dtype=model.dtype).to(model.device) for layer in tqdm(model.model.layers, desc="Awakening predictors...")]
    elif predictor_shape == 'bitlinear':
        return [_make_bitlinear(layer.mlp.up_proj.weight) for layer in tqdm(model.model.layers, desc="Awakening predictors...")]
    else:
        raise NotImplementedError(f"{predictor_shape} is not implemented yet")

def save_predictors(predictors, predictor_shape):
    tensors = {}
    if predictor_shape == 'lowrank':
        for i, (weight_u, weight_v) in enumerate(predictors):
            tensors[f"model.model.layers.{i}.mlp.predictor_u"] = weight_u.contiguous()
            tensors[f"model.model.layers.{i}.mlp.predictor_v"] = weight_v.contiguous()
    elif predictor_shape == 'full':
        for i, weight in enumerate(predictors):
            tensors[f"model.model.layers.{i}.mlp.predictor"] = weight.contiguous()
        else:
            raise NotImplementedError(f"{predictor_shape} is not implemented yet")
    elif predictor_shape == 'kan':
        for i, weight_kan in enumerate(predictors):
            tensors[f"model.model.layers.{i}.mlp.predictor"] = weight_kan
    elif predictor_shape == 'bitlinear':
        for i, weight_bitlinear in enumerate(predictors):
            tensors[f"model.model.layers.{i}.mlp.predictor"] = weight_bitlinear.weight.contiguous()
        
    return tensors  
    