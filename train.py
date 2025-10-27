import os
import gc
import re
import torch
import numpy as np
import argparse
import wandb 
import random
import yaml
import json
import time

from transformers import AutoModelForCausalLM
from copy import deepcopy
from predictrain import init_predictors, countdown_loss, countdown_metrics, flatten_predictors, save_predictors, merge_dicts, preds_and_gts, InverseSqrtScheduler
from tqdm.auto import tqdm
from safetensors.torch import save_file, load_file
from prep_data import DEFAULT_DATASET

from countdown.sparsify import unmerge_phi3_mlp

def time_to_string():
    return time.strftime("%m%d%H%M%S")

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, default='')
argparser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)

argparser.add_argument("--epochs", type=int, default=1)
argparser.add_argument("--train_bsz", type=int, default=1)
argparser.add_argument("--eval_bsz", type=int, default=64)
argparser.add_argument("--lr", type=float, default=2e-5)

argparser.add_argument("--predictor_shape", type=str, default='lowrank')
argparser.add_argument("--predictor_load", type=str, default='')

argparser.add_argument("--countdown_method", type=str, choices=['d-countdown', 'dejavu', 's-countdown'])
argparser.add_argument("--countdown_rank", type=int, default=512)
argparser.add_argument("--countdown_regression", action='store_true')
argparser.add_argument("--countdown_logit_amplify", type=float, default=1.0)
argparser.add_argument("--countdown_logit_amplify_amplify", type=float, default=1.0)
argparser.add_argument("--countdown_learn", type=str, default='classification')
argparser.add_argument("--countdown_sparsity_ratio", type=float, default=0.0)
argparser.add_argument("--countdown_tau", type=float, default=0.0)

argparser.add_argument("--seed", type=int, default=42)
argparser.add_argument("--batch_control", type=str, default='depr')

args = argparser.parse_args()

## set seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

time_key = time_to_string()
    
os.environ['RESULTS_PATH'] = f"./results/countdown_train/{args.model_name.replace('/' , '_')}/{args.countdown_method}/{args.countdown_rank}/{args.countdown_sparsity_ratio}/{args.countdown_learn}/{time_key}"
os.makedirs(os.environ['RESULTS_PATH'], exist_ok=True)
args_dict = vars(args)
with open(f"{os.environ['RESULTS_PATH']}/config.yaml", 'w') as file:
    yaml.dump(args_dict, file)
    
if args.predictor_load:
    tensors = load_file(f"{args.predictor_load}/predictors.safetensors")
    
    predictors = []
    if args.predictor_shape == 'lowrank':
        for i in range(len(tensors) // 2):
            weight_u = tensors[f"model.model.layers.{i}.mlp.predictor_u"].to(device)
            weight_v = tensors[f"model.model.layers.{i}.mlp.predictor_v"].to(device)
            predictors.append((weight_u, weight_v))
    elif args.predictor_shape == 'full':
        for i in range(len(tensors)):
            weight = tensors[f"model.model.layers.{i}.mlp.predictor"].to(device)
            predictors.append(weight)

else:
    assert args.model_name, "Please provide the model name."

    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                attn_implementation="flash_attention_2",
                                                torch_dtype=torch.bfloat16,
                                                cache_dir='./MODELS/').to(device)

    model.train()

    if model.__class__.__name__ == 'Phi3ForCausalLM':
        for idx, layer in tqdm(enumerate(model.model.layers), desc="Unmerging Phi3 layers..."):
            unmerge_phi3_mlp(layer.mlp)

    predictors = init_predictors(model, args.predictor_shape, args.countdown_rank)

    del model

# If there is no trace, we will use the training dataset for calibration format
# Otherwise, we will use the trace dataset for calibration
CALIB_OUTPUT_PATH = f"./DATA/user/countdown/{args.model_name.replace('/' , '__')}/{args.countdown_method}/ideal/0"
CALIB_INPUT_PATH = f"./DATA/user/countdown/{args.model_name.replace('/' , '__')}/commons/ideal/0"
if ',' in args.dataset:
    CALIB_OUTPUT_PATHS = {dataset: f'{CALIB_OUTPUT_PATH}/{dataset}/Train/tensor_states' for dataset in args.dataset.split(',')}
    CALIB_INPUT_PATHS = {dataset: f'{CALIB_INPUT_PATH}/{dataset}/Train/tensor_states' for dataset in args.dataset.split(',')}
    
    for dataset in CALIB_OUTPUT_PATHS:
        assert len(os.listdir(CALIB_OUTPUT_PATHS[dataset])) == len(os.listdir(CALIB_INPUT_PATHS[dataset])), f"Output and input files do not match in {CALIB_OUTPUT_PATHS[dataset]} and {CALIB_INPUT_PATHS[dataset]}"

else :
    CALIB_OUTPUT_PATHS = {args.dataset: (CALIB_OUTPUT_PATH := f'{CALIB_OUTPUT_PATH}/{args.dataset}')}
    CALIB_INPUT_PATHS = {args.dataset: (CALIB_INPUT_PATH := f'{CALIB_INPUT_PATH}/{args.dataset}')}
    
    assert len(os.listdir(CALIB_OUTPUT_PATH)) == len(os.listdir(CALIB_INPUT_PATH)), f"Output and input files do not match in {CALIB_OUTPUT_PATH} and {CALIB_INPUT_PATH}"
        
wandb.init(project='sparsify',
           name=f"{args.model_name.split('/')[-1]}/{args.countdown_method}/{args.countdown_rank}/{args.countdown_sparsity_ratio}/{args.countdown_learn}/{time_key}",
           config=vars(args),
           dir=os.environ['RESULTS_PATH'])

global_step = 0
test_files = [[] for _ in range(len(predictors))]
for i in range(len(predictors)):
    relative_step = 0
    wandb.define_metric(f'layer_{i}_step')  # Define the custom step metric
    wandb.define_metric(f'loss_{i}', step_metric=f'layer_{i}_step')  # Associate loss with custom step
    wandb.define_metric(f'lr_{i}', step_metric=f'layer_{i}_step')    # Associate lr with custom step
    
    calibration_files = list()
    for (i_dset, cal_input_path), (o_dset, cal_output_path) in zip(CALIB_INPUT_PATHS.items(), CALIB_OUTPUT_PATHS.items()):
        assert i_dset == o_dset, f"Input and output datasets do not match: {i_dset} != {o_dset}"
        dset = i_dset
        whole_input_files = [f'{cal_input_path}/{file}' for file in os.listdir(cal_input_path) if re.match(f'calibration_data_layer_{i}_', file)]
        whole_input_files.sort(key=lambda x: int(re.search(r'batch_(\d+).pt', x).group(1)))
        whole_output_files = [f'{cal_output_path}/{file}' for file in os.listdir(cal_output_path) if re.match(f'calibration_data_layer_{i}_', file)]   
        whole_output_files.sort(key=lambda x: int(re.search(r'batch_(\d+).pt', x).group(1)))
        whole_files = list(zip(whole_input_files, whole_output_files))
        num_train = int(len(whole_files) * 0.9)
        calibration_files.extend(whole_files[:-num_train])
        test_files[i].extend(whole_files[-num_train:])
    
    # Make nested list of calibration files
    calibration_files = [calibration_files[i:i+args.train_bsz] for i in range(0, len(calibration_files), args.train_bsz)]
    # Train only when arg.predictor_load is empty
    if args.predictor_load:
        continue
    
    if (args.predictor_shape != 'kan') and (args.predictor_shape != 'bitlinear'):
        optimizer = torch.optim.AdamW(flatten_predictors(predictors[i]), args.lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(predictors[i].parameters(), args.lr, weight_decay=0.01)
    scheduler = InverseSqrtScheduler(optimizer, warmup_steps=(len(calibration_files)) * args.epochs // 10)
    
    predictor = predictors[i]
    
    # Shuffle the calibration files
    random.shuffle(calibration_files)
    for epoch in range(args.epochs):
        for calibration_file in tqdm(calibration_files, desc=f"Training Layer {i} Epoch {epoch}"):
            sparse_inputs_list = []
            origin_logits_list = []

            # Load and concatenate the required trace batches
            for cal_file in calibration_file:
                sparse_input = torch.load(cal_file[0])['sparse_input']
                origin_logit = torch.load(cal_file[1])['origin_logits']

                sparse_inputs_list.append(sparse_input.squeeze())
                origin_logits_list.append(origin_logit.squeeze())
            
            # Concatenate data to form the training batch
            sparse_input = torch.cat(sparse_inputs_list, dim=0).to(device)
            origin_logit = torch.cat(origin_logits_list, dim=0).to(device)
            
            assert sparse_input.shape[0] == origin_logit.shape[0], "Input and output shapes do not match"
            
            optimizer.zero_grad()
            loss = countdown_loss(predictor,
                                sparse_input,
                                origin_logit,
                                args.predictor_shape,
                                args.countdown_tau,
                                args.countdown_sparsity_ratio,
                                args.countdown_regression,
                                args.countdown_logit_amplify if i > 3 else args.countdown_logit_amplify_amplify,
                                args.countdown_learn
                                )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            wandb.log({
                'global_step': global_step,
                'layer': i,
                'relative_step': relative_step,
                f"loss_{i}": loss.item(),
                f"lr_{i}": scheduler.get_last_lr()[0]
            }, step=global_step)
            
            global_step += 1  # Increment the global step
            relative_step += 1  # Increment the relative step for the current layer
                    
            del loss, sparse_input, origin_logit
            torch.cuda.empty_cache()
            gc.collect()
    del optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

test_results = None
cnt = 0

eval_bsz_cnt = 0
sparse_inputs_list = [[] for _ in range(len(predictors))]
origin_logits_list = [[] for _ in range(len(predictors))]

with torch.no_grad():
    for test_batch in tqdm(zip(*test_files), desc="Testing...", total=len(test_files[0])):
        for idx, test_file in enumerate(test_batch):
            sparse_input = torch.load(test_file[0])['sparse_input']
            origin_logit = torch.load(test_file[1])['origin_logits']

            sparse_inputs_list[idx].append(sparse_input.squeeze())
            origin_logits_list[idx].append(origin_logit.squeeze())
            
        eval_bsz_cnt += 1
            
        if eval_bsz_cnt < args.eval_bsz:
            continue

        sparse_inputs_list = [torch.cat(sparse_inputs, dim=0).to(device) for sparse_inputs in sparse_inputs_list]
        origin_logits_list = [torch.cat(origin_logits, dim=0).to(device) for origin_logits in origin_logits_list]
        
        tmp = preds_and_gts(predictors,
                            sparse_inputs_list,
                            origin_logits_list,
                            args.predictor_shape,
                            args.countdown_learn,
                            args.countdown_tau,
                            args.countdown_sparsity_ratio,
                            args.countdown_regression,
                            )
        
        if test_results is None:
            test_results = deepcopy(tmp)
        else:
            test_results = merge_dicts(test_results, deepcopy(tmp), cnt)

        
        sparse_inputs_list = [[] for _ in range(len(predictors))]
        origin_logits_list = [[] for _ in range(len(predictors))]
        
        cnt += 1
        eval_bsz_cnt = 0
        

metrics = dict(sorted(countdown_metrics(test_results).items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999))
metrics.update({k:test_results[k] for k in test_results if 'rho' in k})
metrics.update({k:test_results[k] for k in test_results if 'threshold' in k})
wandb.log({k:metrics[k] for k in metrics if not k.isdigit()})
with open(f"{os.environ['RESULTS_PATH']}/metrics.json", 'w') as file:
    json.dump(metrics, file, indent=4)

wandb.finish()
    
tensors = save_predictors(predictors, predictor_shape=args.predictor_shape)

if args.predictor_shape != 'kan':
    save_file(tensors, f"{os.environ['RESULTS_PATH']}/predictors.safetensors")
else:
    for idx in range(len(tensors)):
        torch.save(tensors[f"model.model.layers.{idx}.mlp.predictor"], f"{os.environ['RESULTS_PATH']}/predictors_{idx}.pt")