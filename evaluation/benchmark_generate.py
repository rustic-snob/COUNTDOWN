import argparse
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from prep_data import lorem_ipsum
from tqdm import tqdm
import yaml
import json
import sys

# Benchmark function
def _benchmark(func, *args, **kwargs):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in tqdm(range(250), desc="Benchmarking"):
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 250  # Average time per run in milliseconds

def benchmark_decode(sparsity_config, output_path, model_name, MAX_LENGTH):
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, "throughput.jsonl")
    os.environ['SAVE_PATH'] = save_path
    os.environ['ABLE_LOG'] = 'false'

    lorem_ipsum_text, _ = lorem_ipsum()
    
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir='./MODELS/', trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    model = countdown_patch(model, sparsity_config)
    
    model.to(device)
    model.eval()
    
    inputs = tokenizer(lorem_ipsum_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    input_ids = inputs['input_ids'][:, :-1].to(device)
    past_key_values = model.forward(input_ids = input_ids, use_cache = True).past_key_values
    
    inputs = tokenizer(lorem_ipsum_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    input_ids = inputs['input_ids'][:, -1:].to(device)
    
    with torch.no_grad():
        for _ in tqdm(range(50), desc="Warming up"):
              model.forward(input_ids = input_ids, past_key_values = past_key_values, use_cache = True)
              
        time_taken = _benchmark(model.forward, input_ids = input_ids, past_key_values = past_key_values, use_cache = True)
    
    print(f"Average time taken for forward pass: {time_taken} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--sparsity_config", type=str, default="")
    parser.add_argument("--output_path", type=str, default="benchmark_generate_test")
    parser.add_argument("--predictor_dir", type=str)
    parser.add_argument("--max_length", type=int, default=256)
    
    args = parser.parse_args()
    if args.sparsity_config:
        sparsity_config = yaml.safe_load(open(args.sparsity_config))
        sparsity_config['predictor_dir'] = args.predictor_dir
        sparsity_config['sp_ideal'] = False
    else:
        sparsity_config = args.sparsity_config

    os.environ["MODEL_NAME"] = args.model_name
    os.environ["NEED_SPEED"] = "True"

    from countdown.sparsify import countdown_patch