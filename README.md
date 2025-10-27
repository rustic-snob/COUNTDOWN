# COUNTDOWN: Contextually Sparse Activation Filtering Out Unnecessary Weights in Down Projection

This repository contains code to reproduce the experiments from our paper "COUNTDOWN: COntextually sparse activation filtering out UNnecessary weighTs in DOWN Projection".

## System Requirements

- Python 3.10+
- CUDA 12.1+ and compatible drivers
- 48GB+ GPU VRAM (24GB+ recommended for larger models)
- Large storage (700GB+ ) for saving trace files of Gated MLP operations

## Setup

```bash
pip install -r requirements.txt
pip install -e .
mkdir -p MODELS DATA results
```

## Folder Structure

```html
COUNTDOWN/
├── countdown/              # Core implementation of COUNTDOWN methods
├── prep_data/              # Data preparation utilities
├── evaluation/             # Evaluation scripts
├── scripts/                # Shell scripts for running experiments
├── config/                 # Configuration files
│   ├── d-countdown/        # Configs for D-COUNTDOWN
│   ├── m-countdown/        # Configs for M-COUNTDOWN
│   ├── cats/               # Configs for CATS baseline
│   └── ...
├── lm-evaluation-harness/  # Downstream task performance evaluation
├── alpaca_eval/            # Chat performance evaluation framework
├── train.py                # Predictor training script
├── requirements.txt        
└── setup.py
```

## Reproducing Paper Results

### Complete Workflow

1. Evaluate SPᴵᵈᵉᵃˡ downstream task performance with `scripts/prob.sh`
2. Train predictor for D-COUNTDOWN with `train.py`
3. Evaluate SPᴾʳᵃᶜ downstream task performance with `scripts/bench_prac.sh`
4. Evaluate chat performance with `scripts/alpaca_gen.sh` and `scripts/alpaca_api.sh` 
5. Run kernel speed benchmarks with `evaluation/benchmark_kernel_test.py`
6. Run E2E speed acceleration measurements with `evaluation/benchmark_generate.py`
7. Generate analysis materials with `evaluation/discussion.py` and `evaluation/benchmark_ternary.py`

### Key Experiment Commands

#### Downstream Task Evaluation (SPᴵᵈᵉᵃˡ)

```bash
# Required parameters:
# MODEL_NAME: Model identifier from HuggingFace
# Optional parameters:
# CUDA_VISIBLE_DEVICES: GPU device ID (default: 0)
   
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" CUDA_VISIBLE_DEVICES=0 bash ./scripts/prob.sh
```

#### Predictor Training

```bash
# Required parameters:
# MODEL_NAME: Model identifier from HuggingFace
# Optional parameters:
# CUDA_VISIBLE_DEVICES: GPU device ID (default: 0)
   
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" CUDA_VISIBLE_DEVICES=0 bash ./scripts/train.sh
```

#### Practical Performance Evaluation (SPᴾʳᵃᶜ)

```bash
# Required parameters:
# MODEL_NAME: Model identifier from HuggingFace
# Optional parameters:
# CUDA_VISIBLE_DEVICES: GPU device ID (default: 0)
   
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" CUDA_VISIBLE_DEVICES=0 bash ./scripts/bench_prac.sh
```

#### Chat Performance Evaluation

```bash
# Step 1: Generate responses
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" CUDA_VISIBLE_DEVICES=0 bash ./scripts/alpaca_gen.sh

# Step 2: Evaluate responses (requires OpenAI API key)
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" OPENAI_API_KEY="your_api_key" bash ./scripts/alpaca_api.sh
```

#### Performance Benchmarking

```bash
# Kernel speed benchmarks
python evaluation/benchmark_kernel_test.py

# End-to-end generation throughput
python evaluation/benchmark_generate.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --sparsity_config config/d-countdown/prac/<<TARGET_CONFIG>>.yaml \
  --max_length 512
```

#### Analysis Data Generation

```bash
# Generate data for visualization and comparison analysis
# This script uses hardcoded model parameters and generates data for the CIF/CAF analysis
python evaluation/discussion.py

# Analyze and benchmark ternary weight patterns as alternative predictors
# Compares performance metrics between ternary weights and low-rank approximation
python evaluation/benchmark_ternary.py
```