import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Any
from collections.abc import Iterable
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple
from torch.optim.lr_scheduler import LambdaLR
    
def _make_binary_sp_ratio(hidden_states, ratio):
    """
    Input:
    hidden_states (N, dim) tensor
    ratio: float value between 0 and 1
    
    Output:
    mask (N, dim) tensor
    threshold (N) tensor
    
    """
    
    # Extract dimensions
    dim = hidden_states.size(-1)

    dtype_hidden_states = hidden_states.dtype
    
    # Calculate the number of elements to zero out per position
    num_to_zero = max(1, int(dim * ratio))  # Scalar value

    # Compute the magnitude (absolute value) of the tensor
    magnitudes = hidden_states.abs()  # Shape: [N, dim]

    # Find the threshold value for each position
    # torch.kthvalue returns the k-th smallest element along the specified dimension
    threshold, _ = torch.kthvalue(magnitudes.to(dtype=torch.float32), k=num_to_zero, dim=-1)  # threshold shape: [N]

    # Expand threshold to match the feature dimension for comparison
    threshold_expanded = threshold.unsqueeze(-1).to(dtype=dtype_hidden_states)  # Shape: [N, 1]
    
    mask = (magnitudes > threshold_expanded)  # Shape: [N, dim]
    
    return mask, threshold.mean().item()

def flatten_predictors(predictors):
    flattened = []
    for item in predictors:
        if isinstance(item, torch.Tensor):
            flattened.append(item)
        elif isinstance(item, Iterable):
            flattened.extend(flatten_predictors(item))
    return flattened
    

def filter_short_samples(example, min_length=50):
    return len(example['text'].split()) >= min_length

def print_parameters(module, title):
    total_params = sum(p.numel() for p in module.parameters())
    print(f"\n{title}:")
    for name, param in module.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")
    print(f"Total parameters: {total_params}")
    print("-------------------")
    
def compare_outputs(original_model, modified_model, input_tensor):
    # Compare the norm of hidden state of the original and modified models
    with torch.no_grad():
        original_output = original_model(**input_tensor, labels=input_tensor['input_ids'], output_hidden_states=True)
        modified_output = modified_model(**input_tensor, labels=input_tensor['input_ids'], output_hidden_states=True)
    difference = torch.norm(original_output['hidden_states'][-1] - modified_output['hidden_states'][-1]).item()
    print(f"original_loss: {original_output['loss']}, modified_loss: {modified_output['loss']}")
    
    return difference
        

def _predictor_logit(sparse_input, predictor, predictor_shape):
    if predictor_shape == 'lowrank':
            # predictor[1] is (hidden_size, r)
            # predictor[0] is (intermediate_size, r)
            # sparse_input is (batch_size, seq_len, hidden_size)
        logit = sparse_input @ predictor[1] @ predictor[0].T
        
    elif predictor_shape == 'full':
        # predictor is (hidden_size, intermediate_size)
        # sparse_input is (batch_size, seq_len, hidden_size)
        logit = sparse_input @ predictor
        
    elif predictor_shape == 'kan':
        logit = predictor(sparse_input)

    elif predictor_shape == 'bitlinear':
        logit = predictor(sparse_input)
        
    return logit

def _predictor_regression(predictor_logit,
                       origin_logit,
                       countdown_learn,
                       countdown_logit_amplify = 1):
    if countdown_learn == 'origin':
        pass
    elif countdown_learn == 'abs':
        origin_logit = torch.abs(origin_logit)
    elif countdown_learn == 'sigmoid':
        origin_logit = torch.sigmoid(origin_logit)
    elif countdown_learn == 'abs_scaled_log':
        origin_logit = torch.log(torch.abs(origin_logit) + 1)
    elif countdown_learn == 'abs_log':
        origin_logit = torch.log(torch.abs(origin_logit))
    else:
        raise NotImplementedError(f"{countdown_learn} is not implemented yet")
    origin_logit = origin_logit * countdown_logit_amplify
    loss = F.mse_loss(predictor_logit, origin_logit)
    
    return loss

def spearmanr_over_dim_pytorch(tensor1, tensor2):
    """
    Computes Spearman's rank correlation coefficient over the 'dim' dimension,

    Parameters:
    - tensor1: Tensor of shape (N, dim)
    - tensor2: Tensor of shape (N, dim)

    Returns:
    - rho: Tensor of shape (N), Spearman's rho for each batch and sequence.
    """
    N, dim = tensor1.shape

    # Rank the data along dim=1 (feature dimension), handling ties
    x_rank = rankdata_torch(tensor1)
    y_rank = rankdata_torch(tensor2)

    # Compute Spearman's rho
    rx_mean = x_rank.mean(dim=1, keepdim=True)
    ry_mean = y_rank.mean(dim=1, keepdim=True)

    numerator = ((x_rank - rx_mean) * (y_rank - ry_mean)).sum(dim=1)
    denominator = torch.sqrt(((x_rank - rx_mean) ** 2).sum(dim=1) * ((y_rank - ry_mean) ** 2).sum(dim=1))

    # Avoid division by zero
    rho_values = numerator / denominator
    rho_values = torch.where(denominator == 0, torch.tensor(float('nan'), device=tensor1.device), rho_values)

    return rho_values.nanmean().item()

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
            
def countdown_loss(predictor,
                sparse_input,
                origin_logit,
                predictor_shape,
                countdown_tau,
                countdown_sparsity_ratio,
                countdown_regression,
                countdown_logit_amplify,
                countdown_learn = 'origin'):  

    total_loss = 0
    
    if countdown_regression:
        logit = _predictor_logit(sparse_input, predictor, predictor_shape)
        
        total_loss += _predictor_regression(logit, origin_logit, countdown_learn, countdown_logit_amplify)
        
    else:
        dtype = sparse_input.dtype
        
        logit = _predictor_logit(sparse_input, predictor, predictor_shape)
        
        label, _ = _make_binary_sp_ratio(origin_logit, countdown_sparsity_ratio)
            
        loss = F.binary_cross_entropy_with_logits(logit,
                                                  label.to(dtype),)        
        total_loss += loss
       
    return total_loss

def preds_and_gts(predictors, sparse_inputs_list, origin_logits_list, predictor_shape, countdown_learn, tau, sparsity_ratio, regression,):
    result = {}
    with torch.no_grad():
        for dec_idx, (predictor, sparse_input, origin_logit) in enumerate(zip(predictors, sparse_inputs_list, origin_logits_list)):
            if regression:
                # Suppose sp_ratio method can only be used in one logit (not tuple)
                sparse_label, _ = _make_binary_sp_ratio(origin_logit, sparsity_ratio)
                logit = _predictor_logit(sparse_input, predictor, predictor_shape)
                
                if 'abs' not in countdown_learn:
                    logit = logit.abs()

                
                rho = spearmanr_over_dim_pytorch(logit.detach().to(torch.float32), origin_logit.abs().to(torch.float32))
                logit, threshold = _make_binary_sp_ratio(logit, sparsity_ratio)
                
                preds = tuple()
                gts = tuple()
                
                preds += ({'unneedy': logit == 0,
                        'needy': logit != 0},)
                gts += ({'unneedy': sparse_label == 0,
                        'needy': sparse_label != 0},)
                
                result[f'rho.{dec_idx}'] = rho
                result[f'threshold.{dec_idx}'] = threshold
            
            else:
                sparse_label, _ = _make_binary_sp_ratio(origin_logit, sparsity_ratio)
                logit = _predictor_logit(sparse_input, predictor, predictor_shape)
                
                preds = tuple()
                gts = tuple()
                
                preds += ({'unneedy': logit <= 0,
                        'needy': logit > 0},)
                gts += ({'unneedy': sparse_label == 0,
                        'needy': sparse_label != 0},)
                
                result[f'threshold.{dec_idx}'] = 0
                
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            
            for prac, gt in zip(preds, gts):
                tp += (prac['needy'] & gt['needy']).sum().item()
                fp += (prac['needy'] & gt['unneedy']).sum().item()
                fn += (prac['unneedy'] & gt['needy']).sum().item()
                tn += (prac['unneedy'] & gt['unneedy']).sum().item()
                
                result[f'tp.{dec_idx}'] = tp
                result[f'fp.{dec_idx}'] = fp
                result[f'fn.{dec_idx}'] = fn
                result[f'tn.{dec_idx}'] = tn
                
            del logit, preds, gts

            torch.cuda.empty_cache()
            gc.collect()
            
    return result   

def merge_dicts(dict1, dict2, num_merged):
    # Merge two dictionaries and sum values of overlapping keys
    result = dict1.copy()  # Start with a copy of the first dictionary
    for key, value in dict2.items():
        if isinstance(value, float):
            if key in result: # take the average of rho values use num_merged
                result[key] = (result[key] * (num_merged) + value) / (num_merged+1)
            else:
                result[key] = value
        if isinstance(value, int):
            if key in result:
                result[key] += value  # Sum values if key is already in the result
            else:
                result[key] = value  # Add new key to the result if not present
        elif isinstance(value, torch.Tensor):
            if key in result:
                result[key] = torch.cat((result[key], value), dim=0)
            else:
                result[key] = value
    return result

def countdown_metrics(test_results: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    metrics = {}
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    
    for layer in set(key.split('.')[1] for key in test_results.keys()):
        try:
            tp = test_results[f'tp.{layer}']
            fp = test_results[f'fp.{layer}']
            fn = test_results[f'fn.{layer}']
            tn = test_results[f'tn.{layer}']
        except KeyError as e:
            print(f"Warning: Missing key {e} for layer {layer}")
            continue
        
        # Calculate total metrics
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        metrics[layer] = _countdown_metrics(tp, fp, fn, tn)

    # Calculate total metrics
    metrics['total'] = _countdown_metrics(total_tp, total_fp, total_fn, total_tn)
    
    return metrics

def _countdown_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }
            
class InverseSqrtScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            
            decay_factor = warmup_steps ** 0.5
            return decay_factor * step ** -0.5

        super(InverseSqrtScheduler, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)
