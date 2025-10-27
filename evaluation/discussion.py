from prep_data import ultrachat
from tqdm.auto import tqdm

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

global_results = pd.DataFrame(columns=["forward_id",
                                       "dec_idx",
                                       "method",
                                       "whitening_ratio",
                                       "CAF_alive",
                                       "CAF_dead",
                                       "CIF_alive",
                                       "CIF_dead",])
act_gate_magnitude = torch.tensor([])
up_magnitude = torch.tensor([])
intermediate_magnitude = torch.tensor([])


def whitening_by_ratio(hidden_states, ratio, abs=False, order_target=None):
    """
    Zero out a portion of elements with the smallest magnitude along the feature dimension for each position.
    
    Args:
        hidden_states (torch.Tensor): Input tensor of shape (Bsz, Seqlen, dim).
        ratio (float): Ratio of elements to zero out (0 < ratio < 1).
        abs (bool): If True, use absolute value for magnitude calculation.
        order_target (torch.Tensor, optional): Alternate tensor for ranking.
    
    Returns:
        pruned_hidden_states (torch.Tensor): Tensor with the smallest elements zeroed.
        threshold_mean (torch.Tensor): Mean threshold value used for pruning.
    """
    Bsz, Seqlen, dim = hidden_states.shape
    dtype_hidden_states = hidden_states.dtype
    num_to_zero = max(1, int(dim * ratio))
    
    if order_target is None:
        order_target = hidden_states.abs() if abs else hidden_states
    else:
        assert order_target.shape == hidden_states.shape, "order_target must have the same shape as hidden_states"
        order_target = order_target.abs() if abs else order_target
    
    threshold, _ = torch.kthvalue(order_target.to(dtype=torch.float32), k=num_to_zero, dim=-1)
    threshold_expanded = threshold.unsqueeze(-1).to(dtype=dtype_hidden_states)
    mask = (order_target > threshold_expanded)
    pruned_hidden_states = hidden_states * mask
    
    return pruned_hidden_states, threshold.mean()

# === Integrated forward method for the MLP layer ===
def forward_mlp_monkey_patch(self, x):
    """
    Integrated forward pass that computes evaluation metrics for all three pruning methods 
    ("cats", "m-countdown", "dejavu") in one go.

    When x.shape[1] != 1, performs a standard forward pass.
    
    In generation mode (x.shape[1]==1), for each method it:
      - Computes the baseline (unpruned) output.
      - Loops over a list of whitening ratios (0.0, 0.1, ..., 0.9), applies the corresponding pruning,
        computes the perturbed output, and measures the L2 vector distance from the baseline.
      - Computes a Spearman rank correlation (comparing an "activation" vector with its filtered version).
      - Records these metrics (with a forward pass id) in a global DataFrame.
    
    Finally, it returns the final output corresponding to the active method (using self.sparsity_ratio).
    """
    global global_results
    global act_gate_magnitude
    global up_magnitude
    global intermediate_magnitude

    # For non-generation phase, do a standard forward.
    if x.shape[1] != 1:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    
    # Maintain a forward counter.
    if not hasattr(self, "forward_count"):
        self.forward_count = 0
    forward_id = self.forward_count
    self.forward_count += 1

    # Predefined list of whitening ratios for evaluation.
    whitening_ratios = [i / 10 for i in range(1, 10)]

    # ---- Compute Baseline Outputs for All Methods ----
    gated_hidden_states = self.gate_proj(x)                                         # "dejavu" branch
    activated_hidden_states = self.act_fn(gated_hidden_states)                      # "cats" branch
    up_projected_states = self.up_proj(x)                                           # "m-countdown" branch
    intermediate_states = activated_hidden_states * up_projected_states             # "d-countdown" branch
    final_down = self.down_proj(intermediate_states)
    
    rows = []

    for ratio in whitening_ratios:
        
        num_alive = int((1 - ratio) * activated_hidden_states.shape[-1])
        num_dead = activated_hidden_states.shape[-1] - num_alive

        # "cats" branch evaluation.
        pruned_cat, _ = whitening_by_ratio(activated_hidden_states, ratio, abs=True)
        
        # "m-countdown" branch evaluation.
        pruned_mcd, _ = whitening_by_ratio(up_projected_states, ratio, abs=True)        
        
        # "d-countdown" branch evaluation.
        pruned_dcount, _ = whitening_by_ratio(intermediate_states, ratio, abs=True)

        d_alive   = pruned_dcount.ne(0)                  # ≡ (pruned_dcount != 0)
        c_alive   = pruned_cat.ne(0)
        m_alive   = pruned_mcd.ne(0)

        num_alive = d_alive.count_nonzero()              # int64 tensor (GPU)
        num_dead  = (~d_alive).count_nonzero()

        # -------- composite masks ------------------
        mcd_fl_1 = (d_alive & ~c_alive)
        mcd_fl_2 = (~d_alive &  c_alive)
        cats_fl_1 = (d_alive & ~m_alive)
        cats_fl_2 = (~d_alive &  m_alive)

        mcd_cr_1 = (mcd_fl_1 &  m_alive)           
        mcd_cr_2 = (mcd_fl_2  & ~m_alive)          
        cats_cr_1 = (cats_fl_1 &  c_alive)          
        cats_cr_2 = (cats_fl_2  & ~c_alive)   

        # counts -> float on CPU *once*, just for the four scalars we actually need
        mcd_cif_alive = (mcd_fl_1.count_nonzero() / num_alive).item()
        mcd_cif_dead  = (mcd_fl_2.count_nonzero() / num_dead).item()
        cats_cif_alive = (cats_fl_1.count_nonzero() / num_alive).item()
        cats_cif_dead  = (cats_fl_2.count_nonzero() / num_dead).item()


        mcd_caf_alive = (mcd_cr_1.count_nonzero() / num_alive).item()
        mcd_caf_dead  = (mcd_cr_2.count_nonzero() / num_dead).item()
        cats_caf_alive = (cats_cr_1.count_nonzero() / num_alive).item()
        cats_caf_dead  = (cats_cr_2.count_nonzero() / num_dead).item()

        # -------- stash two rows in a python list ---
        rows.extend([
            (forward_id, self.dec_idx, "m-countdown", ratio,
            mcd_caf_alive, mcd_caf_dead, mcd_cif_alive, mcd_cif_dead),
            (forward_id, self.dec_idx, "cats", ratio,
            cats_caf_alive, cats_caf_dead, cats_cif_alive, cats_cif_dead),
        ])

    curr_result = pd.DataFrame.from_records(
    rows,
    columns=[
        "forward_id", "dec_idx", "method", "whitening_ratio",
        "CAF_alive", "CAF_dead", "CIF_alive", "CIF_dead",
        ],
    )
    global_results = pd.concat([global_results, curr_result], ignore_index=True)
    
    return final_down

# Monkey-patching function
def monkey_patch(model):        
    # Patch each layer's MLP forward function
    for idx, layer in enumerate(model.model.layers):
        setattr(layer.mlp, 'dec_idx', idx)
        # Monkey-patch the forward method
        layer.mlp.forward = forward_mlp_monkey_patch.__get__(layer.mlp)

    return model

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", cache_dir="./MODELS").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", cache_dir="./MODELS")

model = monkey_patch(model)

samples, _ = ultrachat(10, 0)

generation_config = GenerationConfig(max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)

for sample in tqdm(samples):
    tokenized_chat = tokenizer.apply_chat_template([sample[0]], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(tokenized_chat["input_ids"].to(model.device), attention_mask=tokenized_chat["attention_mask"].to(model.device), generation_config=generation_config)
        print(f"{tokenizer.decode(output[0], skip_special_tokens=True)}")
        print("===")
        del output
        torch.cuda.empty_cache()

# Save the results to a CSV file
global_results.to_csv("discussion.csv", index=False)