import re
import os
import json
import yaml
import torch
import pandas as pd
import time
from tqdm import tqdm

from safetensors.torch import load_file
from functools import wraps
from collections import defaultdict
from types import MethodType

from copy import copy

NEED_SPEED = os.environ.get('NEED_SPEED', None)
if NEED_SPEED is not None:
    print(f"NEED_SPEED is set to {NEED_SPEED}")
    MODEL_NAME = os.environ.get('MODEL_NAME', None)
    assert MODEL_NAME is not None, "MODEL_NAME must be set in the environment variables"

    if 'llama' in MODEL_NAME.lower():
        from ops.kernels.llama_kernels import llama_cats_up as cats_up
        from ops.kernels.llama_kernels import llama_down as mono_down
        from ops.kernels.llama_kernels import llama_mc_up as mc_up
        from ops.kernels.llama_kernels import llama_dcmask_up as dc_up
        from ops.kernels.llama_kernels import llama_dcmask_down as dc_down
    elif 'gemma' in MODEL_NAME.lower():
        from ops.kernels.gemma_kernels import gemma_cats_up as cats_up
        from ops.kernels.gemma_kernels import gemma_down as mono_down
        from ops.kernels.gemma_kernels import gemma_mc_up as mc_up
        from ops.kernels.gemma_kernels import gemma_dcmask_up as dc_up
        from ops.kernels.gemma_kernels import gemma_dcmask_down as dc_down
    elif 'qwen' in MODEL_NAME.lower():
        from ops.kernels.qwen_kernels import qwen_cats_up as cats_up
        from ops.kernels.qwen_kernels import qwen_down as mono_down
        from ops.kernels.qwen_kernels import qwen_mc_up as mc_up
        from ops.kernels.qwen_kernels import qwen_dcmask_up as dc_up
        from ops.kernels.qwen_kernels import qwen_dcmask_down as dc_down
    elif 'phi' in MODEL_NAME.lower():
        from ops.kernels.phi_kernels import phi_cats_up as cats_up
        from ops.kernels.phi_kernels import phi_down as mono_down
        from ops.kernels.phi_kernels import phi_mc_up as mc_up
        from ops.kernels.phi_kernels import phi_dcmask_up as dc_up
        from ops.kernels.phi_kernels import phi_dcmask_down as dc_down
    else:
        raise ValueError(f"MODEL_NAME must contain 'llama', 'gemma', 'qwen' or 'phi' but got {MODEL_NAME}")

CNT = 0

def _weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

# Function to combine directory and dir_type
def trace_dir(sparsity_config , model_name, task_name):
    if model_name and task_name:
        if sparsity_config['state'] == 'ideal':
            directory = os.path.join(sparsity_config['trace_base_dir'], model_name, sparsity_config['method'], sparsity_config['state'], str(sparsity_config['sparsity_ratio']), task_name, sparsity_config['prior_dataset'])
        else:
            directory = os.path.join(sparsity_config['trace_base_dir'], task_name)
    else:
        directory = os.path.join(sparsity_config['trace_base_dir'])
    return directory

def load_predictor(predictor_dir, method, model_dtype, threshold_agg='mean', adjust_threshold = 0.0):
    if method in ['cats', 'm-countdown']:
        dirs = os.listdir(predictor_dir)
        df = pd.DataFrame({'layer': [], 'threshold': []})
        for dir in dirs:
            if not os.path.isfile(full_dir := os.path.join(predictor_dir, dir, 'Train/threshold/threshold_per_layer.jsonl')):
                continue
            df = pd.concat([df, pd.read_json(full_dir, lines=True)])
        if threshold_agg == 'mean':
            threshold_per_layer = df.groupby('layer')['threshold'].mean() + adjust_threshold
        elif threshold_agg == 'median':
            threshold_per_layer = df.groupby('layer')['threshold'].median() + adjust_threshold
        threshold_dict = {k: v for k, v in threshold_per_layer.to_dict().items()} 
        return threshold_dict
    elif method in ['dejavu', 'd-countdown', 's-countdown']:
        loaded_tensor = load_file(f"{predictor_dir}/predictors.safetensors", device='cuda')

        with open (f"{predictor_dir}/config.yaml", 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            should_abs = True if 'origin' in config['countdown_learn'] else False
            predictor_shape = config['predictor_shape']
        
        with open (f"{predictor_dir}/metrics.json", 'r') as f:
            predictor = {int(k.split('.')[1]):{"threshold": float(v) + adjust_threshold, 'should_abs': should_abs} for k,v in json.load(f).items() if k.startswith('threshold')} 

        for key in loaded_tensor.keys():
            layer = int(re.search(r'\d+', key).group())
            name = key.split('mlp.')[1]
            if name.endswith('u'):
                predictor[layer].update({'u': loaded_tensor[key].T.contiguous().to(model_dtype).detach()})
            elif name.endswith('v'):
                predictor[layer].update({'v': loaded_tensor[key].contiguous().to(model_dtype).detach()})
            else:
                # Assign bitlinear instance
                predictor[layer].update({'predictor': torch.nn.Linear(loaded_tensor[key].shape[1], loaded_tensor[key].shape[0]).to('cuda').to(model_dtype)})
                loaded_tensor[key] = _weight_quant(loaded_tensor[key]).contiguous()
                predictor[layer]['predictor'].weight.data = loaded_tensor[key].to(model_dtype).detach()
                predictor[layer]['predictor'].weight.requires_grad = False
            predictor[layer]['predictor_shape'] = predictor_shape
        return predictor
    else:
        raise ValueError(f"method must be one of ['cats', 'dejavu', 'd-countdown', 's-countdown', 'm-countdown'] but got {method}")
    
def calculate_saliency(x: torch.Tensor, W_norm_sq: torch.Tensor) -> torch.Tensor:
    """
    Compute OBD-style saliency for each element of x given W.
    Saliency is defined as the increase in a quadratic 'error':
    
    E_i = 1/2 * x_i^2 * (W^T W)_{ii}
    
    where (W^T W)_{ii} = ||w_i||^2 (the squared norm of the i-th column of W).
    """

    # Saliency_i = 0.5 * x_i^2 * W_norm_sq[i]
    saliency = 0.5 * x.pow(2) * W_norm_sq

    return saliency

def whitening_by_ratio(hidden_states, ratio, abs=False, order_target=None):
    """
    Zero out a portion of elements with the smallest magnitude along the feature dimension for each position.

    Args:
        hidden_states (torch.Tensor): Input tensor of shape (Bsz, Seqlen, dim).
        ratio (float): Ratio of elements to zero out (0 < ratio < 1).
        abs (bool): If True, use the absolute value of the tensor for magnitude calculation.

    Returns:
        torch.Tensor: Tensor with the smallest magnitude elements zeroed out (shape: [Bsz, Seqlen, dim]).
        torch.Tensor: The threshold value used for zeroing elements (shape: [Bsz, Seqlen]).
    """
    # Extract dimensions
    Bsz, Seqlen, dim = hidden_states.shape  # Bsz: batch size, Seqlen: sequence length, dim: feature dimension

    dtype_hidden_states = hidden_states.dtype
    
    # Calculate the number of elements to zero out per position
    num_to_zero = max(1, int(dim * ratio))  # Scalar value

    if order_target is None:
        # Compute the magnitude (absolute value) of the tensor
        order_target = hidden_states.abs() if abs else hidden_states # Shape: [Bsz, Seqlen, dim]
    else:
        assert order_target.shape == hidden_states.shape, "order_target must have the same shape as hidden_states"
        order_target = order_target.abs() if abs else order_target # Shape: [Bsz, Seqlen, dim]

    # Find the threshold value for each position
    # torch.kthvalue returns the k-th smallest element along the specified dimension
    threshold, _ = torch.kthvalue(order_target.to(dtype=torch.float32), k=num_to_zero, dim=-1)  # threshold shape: [Bsz, Seqlen]

    # Expand threshold to match the feature dimension for comparison
    threshold_expanded = threshold.unsqueeze(-1).to(dtype=dtype_hidden_states)  # Shape: [Bsz, Seqlen, 1]

    # Create a mask where elements with magnitude greater than the threshold are kept
    mask = (order_target > threshold_expanded)  # Shape: [Bsz, Seqlen, dim]

    # Apply the mask to zero out the smallest magnitude elements
    pruned_hidden_states = hidden_states * mask  # Shape: [Bsz, Seqlen, dim]

    # Return the pruned tensor and the threshold values
    
    return pruned_hidden_states, threshold.mean()   # pruned_hidden_states shape: [Bsz, Seqlen, dim], threshold shape: [Bsz, Seqlen]

def whitening_by_predictor(inp, predictor, sparsity_ratio=None, method = None, erase_manner='threshold'):
    if isinstance(predictor, float):
        assert method in ['cats', 'm-countdown'], "method must be one of ['cats', 'm-countdown']"
        hidden_states = inp
        hidden_states = torch.where(hidden_states.abs() > predictor,
                                    hidden_states, torch.zeros_like(hidden_states))
        return hidden_states
    elif isinstance(predictor, dict):
        assert method in ['dejavu', 'd-countdown', 's-countdown'], "method must be one of ['dejavu', 'd-countdown', 's-countdown']"
        assert type(predictor['threshold']) == float, "predictor['threshold'] must be a float"
        assert type(predictor['should_abs']) == bool, "predictor['should_abs'] must be a bool"
        
        x = inp
        
        if predictor['predictor_shape'] == 'lowrank':
            u, v = predictor['u'], predictor['v']
            logit = x @ v @ u
        elif predictor['predictor_shape'] == 'bitlinear':
            logit = predictor['predictor'](x)
        
        threshold_expanded = torch.full_like(logit, predictor['threshold'])
        
        if predictor['should_abs']:
            logit = logit.abs()
        
        if erase_manner == 'threshold':
            indices = logit > threshold_expanded
        elif erase_manner == 'topk':
            num_k_elements = int(logit.shape[-1] * (1-sparsity_ratio))
            _, mlp_idx = torch.abs(logit).topk(num_k_elements, dim=-1)
            
            indices = torch.zeros_like(logit, dtype=torch.bool)
            bsz, seq_len, _ = logit.shape
            
            batch_indices = torch.arange(bsz)[:, None, None].expand(bsz, seq_len, num_k_elements)
            seq_indices = torch.arange(seq_len)[None, :, None].expand(bsz, seq_len, num_k_elements)
            
            # Set the top-k positions to True
            indices[batch_indices, seq_indices, mlp_idx] = True
        
        return logit, indices
    else:
        raise ValueError(f"predictor must be a low-rank approximator or a float but got something wrong of: {type(predictor)}")

def spearmanr_over_dim_pytorch(tensor1, tensor2, ignore_mask):
    """
    Computes Spearman's rank correlation coefficient over the 'dim' dimension,
    for each batch and sequence position, excluding positions where ignore_mask == 0.

    Parameters:
    - tensor1: Tensor of shape (bsz, seq_len, dim)
    - tensor2: Tensor of shape (bsz, seq_len, dim)
    - ignore_mask: Tensor of shape (bsz, seq_len), where positions with 0 are ignored.

    Returns:
    - rho: Tensor of shape (bsz, seq_len), Spearman's rho for each batch and sequence.
    """
    bsz, seq_len, dim = tensor1.shape

    # Expand ignore_mask to match the feature dimension
    ignore_mask_expanded = ignore_mask.expand(-1, -1, dim)  # Shape: (bsz, seq_len, dim)

    # Initialize rho tensor with NaNs
    rho = torch.full((bsz, seq_len), float('nan'), dtype=tensor1.dtype, device=tensor1.device)

    # Identify valid positions
    valid_positions = (ignore_mask == 1).nonzero(as_tuple=False)
    if valid_positions.numel() == 0:
        return rho

    # Extract valid data
    b_indices = valid_positions[:, 0]
    s_indices = valid_positions[:, 1]

    x = tensor1[b_indices, s_indices, :]  # Shape: (num_valid_positions, dim)
    y = tensor2[b_indices, s_indices, :]  # Shape: (num_valid_positions, dim)

    # Rank the data along dim=1 (feature dimension), handling ties
    x_rank = rankdata_torch(x)
    y_rank = rankdata_torch(y)

    # Compute Spearman's rho
    rx_mean = x_rank.mean(dim=1, keepdim=True)
    ry_mean = y_rank.mean(dim=1, keepdim=True)

    numerator = ((x_rank - rx_mean) * (y_rank - ry_mean)).sum(dim=1)
    denominator = torch.sqrt(((x_rank - rx_mean) ** 2).sum(dim=1) * ((y_rank - ry_mean) ** 2).sum(dim=1))

    # Avoid division by zero
    rho_values = numerator / denominator
    rho_values = torch.where(denominator == 0, torch.tensor(float('nan'), device=tensor1.device), rho_values)

    # Assign computed rho values to the appropriate positions
    rho[b_indices, s_indices] = rho_values
    
    return rho.nanmean().item()

def rankdata_torch(x):
    """
    Ranks data in x along the last dimension, handling ties by assigning average rank.
    x: Tensor of shape (N, dim)
    Returns: Tensor of ranks with the same shape as x.
    """
    N, dim = x.shape

    # Get the sorted indices and the corresponding sorted data
    sorted_data, sorted_indices = torch.sort(x, dim=1)

    # Initialize a tensor to hold the ranks
    ranks = torch.zeros_like(x, dtype=torch.float32)

    for i in range(N):
        indices = sorted_indices[i]

        # Assign ranks from 1 to dim
        ranks_i = torch.arange(1, dim + 1, dtype=torch.float32, device=x.device)

        # Map ranks back to the original data positions
        ranks[i, indices] = ranks_i

    return ranks

def slice_tensors(x: torch.Tensor, mask: torch.Tensor, hidden: torch.Tensor):
    """
    Slices the input tensors x and hidden based on the mask.

    Args:
        x (torch.Tensor): Tensor of shape (bsz, seqlen, h_dim).
        mask (torch.Tensor): Boolean tensor of shape (bsz, seqlen).
        hidden (torch.Tensor): Tensor of shape (bsz, seqlen, inter_dim).

    Returns:
        x_slice (torch.Tensor): Tensor of shape (N, h_dim) where N is the number of True in mask.
        hidden_slice (torch.Tensor): Tensor of shape (N, inter_dim).
    """
    # Ensure mask is of boolean type
    if mask.dtype != torch.bool:
        mask = mask.bool()
    
    # Use the mask to index the first two dimensions
    # This will flatten the first two dimensions and select where mask is True
    x_slice = x[mask]          # Shape: (N, h_dim)
    hidden_slice = hidden[mask]  # Shape: (N, inter_dim)
    
    return x_slice, hidden_slice

def forward_mlp_monkey_patch(self, x):
    """
    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    """
    if not self.sparse_out_mask:
        assert self.gen, "sparse_out_mask must be provided for non-generative models"
        if x.shape[1] != 1:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            return down_proj
        # breakpoint()
        if self.method == 'cats':
            activated_hidden_states = self.act_fn(self.gate_proj(x))
            if self.state == 'ideal':
                whiten_activated_hidden_states, threshold = whitening_by_ratio(activated_hidden_states,
                                                                            self.sparsity_ratio,
                                                                            abs=True)
                filtered_states = self.up_proj(x) * whiten_activated_hidden_states
                down_projected_states = self.down_proj(filtered_states)
                if not NEED_SPEED:
                    getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'threshold': threshold.item(),
                                                    'sparsity': self.sparsity_ratio})
            elif self.state == 'prac':
                whiten_activated_hidden_states = whitening_by_predictor(activated_hidden_states, self.predictor, self.sparsity_ratio, self.method, self.erase_manner)
                filtered_states = self.up_proj(x) * whiten_activated_hidden_states
                down_projected_states = self.down_proj(filtered_states)
                
                sparsity_ratio = (whiten_activated_hidden_states == 0).sum() / whiten_activated_hidden_states.numel()
                if not NEED_SPEED:
                    getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'sparsity': sparsity_ratio.item(),
                                                    'rho': -100})
        elif self.method == 'm-countdown':
            up_projected_states = self.up_proj(x)
            if self.state == 'ideal':
                whiten_up_projected_states, threshold = whitening_by_ratio(up_projected_states,
                                                                self.sparsity_ratio,
                                                                abs=True)
                down_projected_states = self.down_proj(self.act_fn(self.gate_proj(x)) * whiten_up_projected_states)
                if not NEED_SPEED:
                    getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'threshold': threshold.item(),
                                                    'sparsity': self.sparsity_ratio})
            elif self.state == 'prac':
                whiten_up_projected_states = whitening_by_predictor(up_projected_states, self.predictor, self.sparsity_ratio, self.method, self.erase_manner)
                down_projected_states = self.down_proj(self.act_fn(self.gate_proj(x)) * whiten_up_projected_states)
                
                sparsity_ratio = (whiten_up_projected_states == 0).sum() / whiten_up_projected_states.numel()
                if not NEED_SPEED:
                    getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'sparsity': sparsity_ratio.item(),
                                                    'rho': -100})
        elif self.method == 'dejavu':
            gated_hidden_states = self.gate_proj(x)
            if self.state == 'ideal':
                whiten_gated_hidden_states, threshold = whitening_by_ratio(gated_hidden_states,
                                                                    self.sparsity_ratio,
                                                                    abs=False)
                filtered_states = self.up_proj(x) * self.act_fn(whiten_gated_hidden_states)
                down_projected_states = self.down_proj(filtered_states)
                if not NEED_SPEED:
                    data_to_save= {
                        "sparse_input": x.detach().clone().cpu(),
                        "origin_logits": gated_hidden_states.detach().clone().cpu(),
                    }
                    
                    getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                        'data_to_save': data_to_save,
                                                        'sparsity': self.sparsity_ratio})
            elif self.state == 'prac':
                logits, indices = whitening_by_predictor(x, self.predictor, self.sparsity_ratio, self.method, self.erase_manner)
                whiten_gated_hidden_states = torch.where(indices, gated_hidden_states, torch.zeros_like(gated_hidden_states))
                filtered_states = self.up_proj(x) * self.act_fn(whiten_gated_hidden_states)

                down_projected_states = self.down_proj(filtered_states)
                                
                # rho = spearmanr_over_dim_pytorch(logits.to(torch.float32),
                #                                  gated_hidden_states.to(torch.float32),
                #                                  torch.ones_like(gated_hidden_states))
                
                if not NEED_SPEED:
                    getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'sparsity': self.sparsity_ratio,
                                                    'rho': -100})
        elif self.method == 's-countdown':
            elem_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            if self.state == 'ideal':
                saliency = calculate_saliency(elem_states, self.down_proj_norm_sq)
                whiten_elem_states, threshold = whitening_by_ratio(elem_states,
                                                                self.sparsity_ratio,
                                                                abs=False,
                                                                order_target=saliency)
                down_projected_states = self.down_proj(whiten_elem_states)
                
                data_to_save= {
                    "sparse_input": x.detach().clone().cpu(),
                    "origin_logits": saliency.detach().clone().cpu(),
                }
                
                getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'data_to_save': data_to_save,
                                                    'sparsity': self.sparsity_ratio})
            elif self.state == 'prac':
                logits, indices = whitening_by_predictor(x, self.predictor, self.sparsity_ratio, self.method, self.erase_manner)
                whiten_elem_states = torch.where(indices, elem_states, torch.zeros_like(whiten_elem_states))
                down_projected_states = self.down_proj(whiten_elem_states)
                
                sparsity_ratio = (whiten_elem_states == 0).sum() / whiten_elem_states.numel()
                
                getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'sparsity': sparsity_ratio.item(),
                                                    'rho': -100})
                
        elif self.method == 'd-countdown':
            elem_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            if self.state == 'ideal':
                whiten_elem_states, threshold = whitening_by_ratio(elem_states,
                                                                self.sparsity_ratio,
                                                                abs=True)
                down_projected_states = self.down_proj(whiten_elem_states)
                if not NEED_SPEED:
                    data_to_save= {
                        "sparse_input": x.detach().clone().cpu(),
                        "origin_logits": elem_states.detach().clone().cpu(),
                    }
                    
                    getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                        'data_to_save': data_to_save,
                                                        'sparsity': self.sparsity_ratio})
            elif self.state == 'prac':
                logits, indices = whitening_by_predictor(x, self.predictor, self.sparsity_ratio, self.method, self.erase_manner)
                whiten_elem_states = torch.where(indices, elem_states, torch.zeros_like(elem_states))
                down_projected_states = self.down_proj(whiten_elem_states)
                
                sparsity_ratio = (whiten_elem_states == 0).sum() / whiten_elem_states.numel()
                                
                # rho = spearmanr_over_dim_pytorch(logits.to(torch.float32),
                #                                  elem_states.abs().to(torch.float32),
                #                                  torch.ones_like(elem_states))
                if not NEED_SPEED:
                    getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                        'sparsity': sparsity_ratio.item(),
                                                        'rho': -100})
        elif self.method == 'trace_only':
            up_projected_states = self.up_proj(x)
            gated_hidden_states = self.gate_proj(x)
            activated_hidden_states = self.act_fn(gated_hidden_states)
            elem_states = activated_hidden_states * up_projected_states
            # saliency = calculate_saliency(elem_states, self.down_proj_norm_sq)
            down_projected_states = self.down_proj(elem_states)
            if self.state == 'ideal':
                data_to_save = {"commons": {"sparse_input": x.detach().clone().cpu(),},
                                # "dejavu": {"origin_logits": gated_hidden_states.detach().clone().cpu()},
                                "d-countdown": {"origin_logits": elem_states.detach().clone().cpu()},
                                # "s-countdown": {"origin_logits": saliency.detach().clone().cpu()},
                                }
                
                getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'data_to_save': data_to_save,})
    else:
        sparse_out_mask = self.sparse_out_mask[0]
        assert not self.gen, "sparse_out_mask must not be provided for generative models"
        if self.method == 'cats':
            activated_hidden_states = self.act_fn(self.gate_proj(x))
            sparse_out_mask_expanded = sparse_out_mask.unsqueeze(-1).expand_as(activated_hidden_states)
            if self.state == 'ideal':
                whiten_activated_hidden_states, threshold = whitening_by_ratio(activated_hidden_states,
                                                                            self.sparsity_ratio,
                                                                            abs=True)
                whiten_activated_hidden_states = torch.where(sparse_out_mask_expanded, whiten_activated_hidden_states, activated_hidden_states)
                filtered_states = self.up_proj(x) * whiten_activated_hidden_states
                down_projected_states = self.down_proj(filtered_states)
                
                getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'threshold': threshold.item(),
                                                    'sparsity': self.sparsity_ratio})
            elif self.state == 'prac':
                whiten_activated_hidden_states = whitening_by_predictor(activated_hidden_states, self.predictor, self.sparsity_ratio, self.method, self.erase_manner)
                whiten_activated_hidden_states = torch.where(sparse_out_mask_expanded, whiten_activated_hidden_states, activated_hidden_states)
                filtered_states = self.up_proj(x) * whiten_activated_hidden_states
                down_projected_states = self.down_proj(filtered_states)
                # sparsity ratio calculated on positions where sparse_out_mask == 1
                sparsity_ratio = torch.sum(sparse_out_mask_expanded * (whiten_activated_hidden_states == 0)) / torch.sum(sparse_out_mask_expanded)  
                
                getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'sparsity': sparsity_ratio.item(),
                                                    'rho': 1})
        elif self.method == 'm-countdown':
            up_projected_states = self.up_proj(x)
            sparse_out_mask_expanded = sparse_out_mask.unsqueeze(-1).expand_as(up_projected_states)
            if self.state == 'ideal':
                whiten_up_projected_states, threshold = whitening_by_ratio(up_projected_states,
                                                                self.sparsity_ratio,
                                                                abs=True)
                whiten_up_projected_states = torch.where(sparse_out_mask_expanded, whiten_up_projected_states, up_projected_states)
                down_projected_states = self.down_proj(self.act_fn(self.gate_proj(x)) * whiten_up_projected_states)
                
                
                getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'threshold': threshold.item(),
                                                    'sparsity': self.sparsity_ratio})
            elif self.state == 'prac':
                whiten_up_projected_states = whitening_by_predictor(up_projected_states, self.predictor, self.sparsity_ratio, self.method, self.erase_manner)
                whiten_up_projected_states = torch.where(sparse_out_mask_expanded, whiten_up_projected_states, up_projected_states)
                down_projected_states = self.down_proj(self.act_fn(self.gate_proj(x)) * whiten_up_projected_states)

                sparsity_ratio = torch.sum(sparse_out_mask_expanded * (whiten_up_projected_states == 0)) / torch.sum(sparse_out_mask_expanded)
                
                getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'sparsity': sparsity_ratio.item(),
                                                    'rho': 1})
        elif self.method == 'dejavu':
            gated_hidden_states = self.gate_proj(x)
            sparse_out_mask_expanded = sparse_out_mask.unsqueeze(-1).expand_as(gated_hidden_states)
            if self.state == 'ideal':
                whiten_gated_hidden_states, threshold = whitening_by_ratio(gated_hidden_states,
                                                                    self.sparsity_ratio,
                                                                    abs=False)
                whiten_gated_hidden_states = torch.where(sparse_out_mask_expanded, whiten_gated_hidden_states, gated_hidden_states)
                filtered_states = self.up_proj(x) * self.act_fn(whiten_gated_hidden_states)
                down_projected_states = self.down_proj(filtered_states)
                
                x_slice, hidden_slice = slice_tensors(x, sparse_out_mask, gated_hidden_states)
                
                data_to_save= {
                    "sparse_input": x_slice.detach().clone().cpu(),
                    "origin_logits": hidden_slice.detach().clone().cpu(),
                }
                
                getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'data_to_save': data_to_save,
                                                    'sparsity': self.sparsity_ratio})
            elif self.state == 'prac':
                logits, indices = whitening_by_predictor(x, self.predictor, self.sparsity_ratio, self.method, self.erase_manner)
                whiten_gated_hidden_states = torch.where(indices | ~sparse_out_mask_expanded, gated_hidden_states, torch.zeros_like(gated_hidden_states))
                filtered_states = self.up_proj(x) * self.act_fn(whiten_gated_hidden_states)
                down_projected_states = self.down_proj(filtered_states)
                
                getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'sparsity': self.sparsity_ratio,
                                                    'rho': -100})
        elif self.method == 's-countdown':
            elem_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            sparse_out_mask_expanded = sparse_out_mask.unsqueeze(-1).expand_as(elem_states)
            if self.state == 'ideal':
                saliency = calculate_saliency(elem_states, self.down_proj_norm_sq)
                whiten_elem_states, threshold = whitening_by_ratio(elem_states,
                                                                self.sparsity_ratio,
                                                                abs=False,
                                                                order_target=saliency)
                whiten_elem_states = torch.where(sparse_out_mask_expanded, whiten_elem_states, elem_states)
                down_projected_states = self.down_proj(whiten_elem_states)
                
                data_to_save= {
                    "sparse_input": x.detach().clone().cpu(),
                    "origin_logits": saliency.detach().clone().cpu(),
                }
                
                getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                        'data_to_save': data_to_save,
                                                        'sparsity': self.sparsity_ratio})
            elif self.state == 'prac':
                logits, indices = whitening_by_predictor(x, self.predictor, self.sparsity_ratio, self.method, self.erase_manner)
                whiten_elem_states = torch.where(indices | ~sparse_out_mask_expanded, elem_states, torch.zeros_like(elem_states))
                down_projected_states = self.down_proj(whiten_elem_states)
                
                sparsity_ratio = torch.sum(sparse_out_mask_expanded * (whiten_elem_states == 0)) / torch.sum(sparse_out_mask_expanded)
                
                getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                        'sparsity': sparsity_ratio.item(),
                                                        'rho': -100})
            
        elif self.method == 'd-countdown':
            elem_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            sparse_out_mask_expanded = sparse_out_mask.unsqueeze(-1).expand_as(elem_states)
            if self.state == 'ideal':
                whiten_elem_states, threshold = whitening_by_ratio(elem_states,
                                                                self.sparsity_ratio,
                                                                abs=True)
                whiten_elem_states = torch.where(sparse_out_mask_expanded, whiten_elem_states, elem_states)
                down_projected_states = self.down_proj(whiten_elem_states)
                
                x_slice, hidden_slice = slice_tensors(x, sparse_out_mask, elem_states)
                
                data_to_save= {
                    "sparse_input": x_slice.detach().clone().cpu(),
                    "origin_logits": hidden_slice.detach().clone().cpu(),
                }
                
                getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                        'data_to_save': data_to_save,
                                                        'sparsity': self.sparsity_ratio})
            elif self.state == 'prac':
                logits, indices = whitening_by_predictor(x, self.predictor, self.sparsity_ratio, self.method, self.erase_manner)
                whiten_elem_states = torch.where(indices | ~sparse_out_mask_expanded, elem_states, torch.zeros_like(elem_states))
                down_projected_states = self.down_proj(whiten_elem_states)
                
                sparsity_ratio = torch.sum(sparse_out_mask_expanded * (whiten_elem_states == 0)) / torch.sum(sparse_out_mask_expanded)
                
                getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'sparsity': sparsity_ratio.item(),
                                                    'rho': -100})
        elif self.method == 'trace_only':
            up_projected_states = self.up_proj(x)
            gated_hidden_states = self.gate_proj(x)
            activated_hidden_states = self.act_fn(gated_hidden_states)
            elem_states = activated_hidden_states * up_projected_states
            # saliency = calculate_saliency(elem_states, self.down_proj_norm_sq)
            down_projected_states = self.down_proj(elem_states)
            
            # _, dejavu_slice = slice_tensors(x, sparse_out_mask, gated_hidden_states)
            x_slice, d_countdown_slice = slice_tensors(x, sparse_out_mask, elem_states)
            # _, s_countdown_slice = slice_tensors(x, sparse_out_mask, saliency)
            
            if self.state == 'ideal':
                data_to_save = {"commons": {"sparse_input": x_slice.detach().clone().cpu(),},
                                #  "dejavu": {"origin_logits": dejavu_slice.detach().clone().cpu()},
                                "d-countdown": {"origin_logits": d_countdown_slice.detach().clone().cpu()},
                                #  "s-countdown": {"origin_logits": s_countdown_slice.detach().clone().cpu()},
                                }
                
                getattr(self, 'trace_dicts').append({'layer': self.dec_idx,
                                                    'data_to_save': data_to_save,})
            
    return down_projected_states

def _forward(forward_fn):
    @wraps(forward_fn)
    def wrapped_forward(*args, **kwargs):
        self = args[0]
        # Extract 'sparse_out_mask' from kwargs, if it exists
        sparse_out_mask = kwargs.pop('sparse_out_mask', None)

        # If present, store it as an attribute on the model
        if sparse_out_mask is not None:
            self.sparse_out_mask.clear()
            self.sparse_out_mask.append(sparse_out_mask)

        # Call the original forward method with remaining args/kwargs
        
        output = forward_fn(**kwargs)
        if NEED_SPEED:
            return output
        for idx, layer_dict in enumerate(self.trace_dicts):
            for artifact, value in layer_dict.items():
                if artifact == 'layer':
                    continue
                elif (artifact == 'threshold') and self.state == 'ideal':
                    file_name = f"{artifact}_per_layer.jsonl"
                    file_path = os.path.join(self.trace_dir, artifact, file_name)
                    with open(file_path, 'a') as f:
                        json.dump({'layer': idx, artifact: value}, f)
                        f.write('\n')
                elif (artifact == 'sparsity') and self.state == 'prac':
                    file_name = f"{artifact}_per_layer.jsonl"
                    file_path = os.path.join(self.trace_dir, artifact, file_name)
                    with open(file_path, 'a') as f:
                        json.dump({'layer': idx, artifact: value}, f)
                        f.write('\n')
                elif artifact == 'data_to_save':
                    if self.save_tensors:
                        if self.method == 'trace_only':
                            for cur_method, sub_value in value.items():
                                # Change 'trace_only' to cur_method in the file path
                                cur = re.sub('trace_only', cur_method, self.trace_dir)
                                os.makedirs(os.path.join(cur, 'tensor_states'), exist_ok=True)
                                cur_file_path = os.path.join(cur, "tensor_states", f"calibration_data_layer_{idx}_batch_{self.trace_cnt}.pt")
                                # check if the file exists, save only if it does not
                                if not os.path.exists(cur_file_path):
                                    torch.save(sub_value, cur_file_path)
                                else:
                                    pass
                        else:
                            file_path = os.path.join(self.trace_dir, "tensor_states", f"calibration_data_layer_{idx}_batch_{self.trace_cnt}.pt")
                            torch.save(value, file_path)
        self.trace_dicts.clear()
        self.trace_cnt += 1
        
        return output
    
    return wrapped_forward

def unmerge_phi3_mlp(mlp):
    """
    Unmerges the gate_up_proj weights into separate gate_proj and up_proj weights.
    """
    # Get the existing merged gate_up_proj weights
    merged_weight = mlp.gate_up_proj.weight  # Shape: [2 * intermediate_size, hidden_size]

    # Split the weights
    intermediate_size = mlp.config.intermediate_size
    gate_weight = merged_weight[:intermediate_size, :]
    up_weight = merged_weight[intermediate_size:, :]

    # Create new separate linear layers
    mlp.gate_proj = torch.nn.Linear(mlp.config.hidden_size, intermediate_size, bias=False, dtype=mlp.gate_up_proj.weight.dtype, device=mlp.gate_up_proj.weight.device)
    mlp.up_proj = torch.nn.Linear(mlp.config.hidden_size, intermediate_size, bias=False, dtype=mlp.gate_up_proj.weight.dtype, device=mlp.gate_up_proj.weight.device)

    # Assign the split weights
    mlp.gate_proj.weight.data.copy_(gate_weight)
    mlp.up_proj.weight.data.copy_(up_weight)

    mlp.hidden_size = mlp.config.hidden_size
    mlp.intermediate_size = mlp.config.intermediate_size
    mlp.act_fn = copy(mlp.activation_fn)

    del mlp.gate_up_proj, mlp.activation_fn


# Monkey-patching function
def monkey_patch(model, sparsity_config, model_name=None, task_name=None):
    model.trace_dir = trace_dir(sparsity_config, model_name, task_name)
    os.makedirs(os.path.join(model.trace_dir, 'threshold'), exist_ok=True)
    os.makedirs(os.path.join(model.trace_dir, 'sparsity'), exist_ok=True)
    os.makedirs(os.path.join(model.trace_dir, 'tensor_states'), exist_ok=True)
    # asserts that all of three folders are empty
    # assert len(os.listdir(os.path.join(model.trace_dir, 'threshold'))) == 0
    # assert len(os.listdir(os.path.join(model.trace_dir, 'sparsity'))) == 0
    # assert len(os.listdir(os.path.join(model.trace_dir, 'tensor_states'))) == 0
    
    model.trace_dicts = list()
    model.trace_cnt = 0
    model.predictor = None
    
    model.state = sparsity_config.get('state', 'ideal' if sparsity_config.get('sp_ideal', None) else 'prac')
    model.method = sparsity_config.get('method', None)
    model.gen = sparsity_config.get('gen', None)
    model.sparsity_ratio = sparsity_config.get('sparsity_ratio', None)
    model.save_tensors = sparsity_config.get('save_tensors', None)
    model.prior_dataset = sparsity_config.get('prior_dataset', None)
    model.erase_manner = sparsity_config.get('erase_manner', 'threshold')

    if model.state == 'prac':
        model.predictor = load_predictor(sparsity_config['predictor_dir'],
                                        model.method,
                                        model.dtype,
                                        sparsity_config.get('threshold_aggregate', 'mean'),
                                        sparsity_config.get('adjust_threshold', 0.0),)

    model.sparse_out_mask = list()
    
    # Patch each layer's MLP forward function
    for idx, layer in tqdm(enumerate(model.model.layers), desc="Patching layers", total=len(model.model.layers)):
        if model.__class__.__name__ == 'Phi3ForCausalLM':
            unmerge_phi3_mlp(layer.mlp)
        setattr(layer.mlp, 'dec_idx', idx)
        setattr(layer.mlp, 'state', model.state)
        setattr(layer.mlp, 'method', model.method)
        setattr(layer.mlp, 'gen', model.gen)
        setattr(layer.mlp, 'sparse_out_mask', model.sparse_out_mask)
        setattr(layer.mlp, 'trace_dicts', model.trace_dicts)
        setattr(layer.mlp, 'sparsity_ratio', model.sparsity_ratio)
        if model.method in [
            's-countdown',
            # 'trace_only',
            ]:
            setattr(layer.mlp, 'down_proj_norm_sq', torch.norm(layer.mlp.down_proj.weight, p=2, dim=0).pow(2))
        if model.predictor is not None:
            setattr(layer.mlp, 'erase_manner', model.erase_manner)
            setattr(layer.mlp, 'predictor', model.predictor[idx])
        # Monkey-patch the forward method
        layer.mlp.forward = forward_mlp_monkey_patch.__get__(layer.mlp)

    # Patch the model's forward method     
    model.forward = MethodType(_forward(model.forward), model)
    
    return model

def gemv_gemv_triton(x, x_1, Wup, Wdownt, threshold):
    flags = torch.abs(x_1) > threshold
    x = cats_up(x, x_1, Wup, flags)
    return mono_down(x, Wdownt, flags)

def m_countdown_gemv_gemv_triton(x, x_1, Wgate, Wdownt, threshold):
    flags = torch.abs(x_1) > threshold
    x = mc_up(x, x_1, Wgate, flags)
    return mono_down(x, Wdownt, flags)

def method_sparse_prediction_unified(x, predictor_u, predictor_v, threshold=0):
    x = x.squeeze()
    prac = (x @ predictor_v @ predictor_u)
    column_used = (prac > threshold)
    return torch.nonzero(column_used)

@torch.jit.script
def method_sparse_prediction_bitlinear(x: torch.Tensor,
                                    weight: torch.Tensor,  # (out, in)
                                    threshold: float = 0.0):
    x = x.squeeze()
    prac = torch.matmul(x, weight.t())      # = weight(x)
    column_used = prac > threshold
    return torch.nonzero(column_used)

def fused_triton_sparsed(x, w_gate, w_up, w_down, sparse_index):
    return dc_down(dc_up(x, w_gate,
                                                                                        w_up,
                                                                                        sparse_index),
                                        w_down, sparse_index)

def forward_mlp_countdown_patch(self, x):
    if x.shape[1] != 1:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    if self.state == 'prac':
        if x.shape[0] == 1:    
            if self.method == 'cats':
                down_projected_states = gemv_gemv_triton(x,
                                                        self.act_fn(self.gate_proj(x)),
                                                        self.up_proj.weight,
                                                        self.down_t_proj,
                                                        self.predictor)
            elif self.method == 'm-countdown':
                down_projected_states = m_countdown_gemv_gemv_triton(x,
                                                                    self.up_proj(x),
                                                                    self.gate_proj.weight,
                                                                    self.down_t_proj,
                                                                    self.predictor)
                
            elif self.method == 'd-countdown':
                try:
                    down_projected_states = fused_triton_sparsed(x, self.gate_proj.weight, self.up_proj.weight, self.down_t_proj, method_sparse_prediction_unified(x, self.predictor['u'], self.predictor['v'], self.predictor['threshold']))
                except:
                    down_projected_states = fused_triton_sparsed(x, self.gate_proj.weight, self.up_proj.weight, self.down_t_proj, method_sparse_prediction_bitlinear(x, self.predictor['predictor'].weight, self.predictor['threshold']))
                
            elif self.method == 'dejavu':
                NotImplementedError("dejavu is not implemented yet")
                
            elif self.method == 's-countdown':
                down_projected_states = fused_triton_sparsed(x, self.gate_proj.weight, self.up_proj.weight, self.down_t_proj, method_sparse_prediction_unified(x, self.predictor['u'], self.predictor['v']))
            elif self.method == 'dense':
                down_projected_states = torch.matmul((self.act_fn(torch.matmul(x, self.gate_proj.weight)) * torch.matmul(x, self.up_proj.weight)), self.down_t_proj)
        else:
            if self.method == 'cats':
                activated_hidden_states = self.act_fn(self.gate_proj(x))
                whiten_activated_hidden_states = whitening_by_predictor(activated_hidden_states, self.predictor, None, self.method, None)
                filtered_states = self.up_proj(x) * whiten_activated_hidden_states
                down_projected_states = self.down_proj(filtered_states)

            elif self.method == 'm-countdown':
                up_projected_states = self.up_proj(x)
                whiten_up_projected_states = whitening_by_predictor(up_projected_states, self.predictor, None, self.method, None)
                down_projected_states = self.down_proj(self.act_fn(self.gate_proj(x)) * whiten_up_projected_states)

            elif self.method == 'd-countdown':
                elem_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                logits, indices = whitening_by_predictor(x, self.predictor, None, self.method, "threshold")
                whiten_elem_states = torch.where(indices, elem_states, torch.zeros_like(elem_states))
                down_projected_states = self.down_proj(whiten_elem_states)

            elif self.method == 'dejavu':
                NotImplementedError("dejavu is not implemented yet")

    elif self.state == 'ideal':
        if self.method == 'cats':
            activated_hidden_states = self.act_fn(self.gate_proj(x))
            whiten_activated_hidden_states, threshold = whitening_by_ratio(activated_hidden_states,
                                                                        self.sparsity_ratio,
                                                                        abs=True)
            filtered_states = self.up_proj(x) * whiten_activated_hidden_states
            down_projected_states = self.down_proj(filtered_states)
            
        elif self.method == 'm-countdown':
            up_projected_states = self.up_proj(x)
            whiten_up_projected_states, threshold = whitening_by_ratio(up_projected_states,
                                                            self.sparsity_ratio,
                                                            abs=True)
            down_projected_states = self.down_proj(self.act_fn(self.gate_proj(x)) * whiten_up_projected_states)
        
        elif self.method == 'd-countdown':
            elem_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            whiten_elem_states, threshold = whitening_by_ratio(elem_states,
                                                            self.sparsity_ratio,
                                                            abs=True)
            down_projected_states = self.down_proj(whiten_elem_states)
            
        elif self.method == 'dejavu':
            NotImplementedError("dejavu is not implemented yet")
            
        elif self.method == 's-countdown':
            elem_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            saliency = calculate_saliency(elem_states, self.down_proj_norm_sq)
            whiten_elem_states, threshold = whitening_by_ratio(elem_states,
                                                            self.sparsity_ratio,
                                                            abs=False,
                                                            order_target=saliency)
            down_projected_states = self.down_proj(whiten_elem_states)
        elif self.method == 'dense':
            down_projected_states = torch.matmul((self.act_fn(torch.matmul(x, self.gate_t_proj)) * torch.matmul(x, self.up_t_proj)), self.down_t_proj)
        
    elif self.method == 'dense':
        down_projected_states = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    return down_projected_states

def countdown_patch(model, sparsity_config):
    model.predictor = None
    device = model.device
    if sparsity_config:
        if sparsity_config['sp_ideal']:
            model.method = sparsity_config['method']
            model.state = 'ideal'
            model.sparsity_ratio = sparsity_config['sparsity_ratio']
        else:
            model.method = sparsity_config['method']
            model.state = 'prac'
            model.sparsity_ratio = None
            model.predictor = load_predictor(sparsity_config['predictor_dir'],
                                            model.method,
                                            model.dtype,
                                            'mean',
                                            sparsity_config.get('adjust_threshold', 0.0))
    else:
        model.method = 'dense'
        model.state = 'ideal'
        model.sparsity_ratio = None
        model.predictor = [None] * len(model.model.layers)
    
    # Patch each layer's MLP forward function
    for idx, layer in tqdm(enumerate(model.model.layers), desc="Patching layers", total=len(model.model.layers)):
        setattr(layer.mlp, 'dec_idx', idx)
        setattr(layer.mlp, 'method', model.method)
        setattr(layer.mlp, 'state', model.state)
        setattr(layer.mlp, 'sparsity_ratio', model.sparsity_ratio)
        if model.predictor is not None:
            setattr(layer.mlp, 'predictor', model.predictor[idx])
        setattr(layer.mlp, 'down_t_proj', layer.mlp.down_proj.weight.detach().t().contiguous().to(device))
        if model.method == 'dense':
            setattr(layer.mlp, 'up_t_proj', layer.mlp.up_proj.weight.detach().t().contiguous().to(device))
            setattr(layer.mlp, 'gate_t_proj', layer.mlp.gate_proj.weight.detach().t().contiguous().to(device))
        layer.mlp.forward = forward_mlp_countdown_patch.__get__(layer.mlp)    

    return model