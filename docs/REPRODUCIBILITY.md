# Reproducibility Guide

This document provides detailed instructions for reproducing the results from the Prover-Verifier Games paper.

## Environment Setup

### 1. System Requirements

- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher (for GPU training)
- **Memory**: At least 32GB RAM (64GB recommended)
- **Storage**: At least 100GB free space
- **GPU**: NVIDIA GPU with at least 16GB VRAM (24GB+ recommended)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/pvg-legibility.git
cd pvg-legibility

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 3. Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=.cache/transformers
export HF_HOME=.cache/huggingface
```

## Data Preparation

### 1. Download GSM8K Dataset

```bash
python scripts/prepare_gsm8k.py --output-dir data/gsm8k --verbose
```

### 2. Create Subsets (Optional)

```bash
# Create a 1000-sample subset for quick testing
python scripts/prepare_gsm8k.py --output-dir data/gsm8k --subset 1000 --verbose
```

## Model Training

### 1. Single Round Training

```bash
# Train with default configuration
python scripts/run_round.py --config configs/default.yaml --data-dir data/gsm8k --output-dir outputs/round_001 --device cuda --verbose

# Train with specific models
python scripts/run_round.py \
  --config configs/default.yaml \
  --data-dir data/gsm8k \
  --output-dir outputs/round_001 \
  --device cuda \
  --verbose
```

### 2. Multi-Round Training

```bash
# Train for multiple rounds
for round in {1..5}; do
  python scripts/run_round.py \
    --config configs/default.yaml \
    --data-dir data/gsm8k \
    --output-dir outputs/round_$(printf "%03d" $round) \
    --device cuda \
    --verbose
done
```

### 3. Experiment Configurations

```bash
# SRC reward experiment
python scripts/run_round.py --config configs/experiments.yaml:src_experiment --data-dir data/gsm8k --output-dir outputs/src_experiment --device cuda

# CGC reward experiment
python scripts/run_round.py --config configs/experiments.yaml:cgc_experiment --data-dir data/gsm8k --output-dir outputs/cgc_experiment --device cuda

# Goodharting experiment
python scripts/run_round.py --config configs/experiments.yaml:goodhart_experiment --data-dir data/gsm8k --output-dir outputs/goodhart_experiment --device cuda
```

## Evaluation

### 1. Comprehensive Evaluation

```bash
# Evaluate trained models
python scripts/eval_all.py \
  --model-dir outputs/round_001 \
  --data-dir data/gsm8k \
  --output-dir outputs/eval_001 \
  --device cuda \
  --verbose
```

### 2. Attack Evaluation

```bash
# Test sneaky-only attacks
python scripts/attack_sneaky_only.py \
  --model-dir outputs/round_001 \
  --data-dir data/gsm8k \
  --output-dir outputs/attacks_001 \
  --reward goodhart \
  --max-attempts 10 \
  --device cuda \
  --verbose
```

## Reproducing Paper Results

### 1. Main Results (Figure 1)

```bash
# Train with SRC reward (default)
python scripts/run_round.py --config configs/default.yaml --data-dir data/gsm8k --output-dir outputs/src_main --device cuda

# Train with CGC reward
python scripts/run_round.py --config configs/experiments.yaml:cgc_experiment --data-dir data/gsm8k --output-dir outputs/cgc_main --device cuda

# Train with Goodharting
python scripts/run_round.py --config configs/experiments.yaml:goodhart_experiment --data-dir data/gsm8k --output-dir outputs/goodhart_main --device cuda

# Evaluate all
python scripts/eval_all.py --model-dir outputs/src_main --data-dir data/gsm8k --output-dir outputs/eval_src_main --device cuda
python scripts/eval_all.py --model-dir outputs/cgc_main --data-dir data/gsm8k --output-dir outputs/eval_cgc_main --device cuda
python scripts/eval_all.py --model-dir outputs/goodhart_main --data-dir data/gsm8k --output-dir outputs/eval_goodhart_main --device cuda
```

### 2. Best-of-n Evaluation (Figure 2)

```bash
# Train multiple rounds
for round in {1..5}; do
  python scripts/run_round.py \
    --config configs/default.yaml \
    --data-dir data/gsm8k \
    --output-dir outputs/round_$(printf "%03d" $round) \
    --device cuda
done

# Evaluate best-of-n for each round
for round in {1..5}; do
  python scripts/eval_all.py \
    --model-dir outputs/round_$(printf "%03d" $round) \
    --data-dir data/gsm8k \
    --output-dir outputs/eval_round_$(printf "%03d" $round) \
    --device cuda
done
```

### 3. Attack Evaluation (Figure 3)

```bash
# Test attacks against different reward functions
python scripts/attack_sneaky_only.py --model-dir outputs/src_main --data-dir data/gsm8k --output-dir outputs/attacks_src --reward src --device cuda
python scripts/attack_sneaky_only.py --model-dir outputs/cgc_main --data-dir data/gsm8k --output-dir outputs/attacks_cgc --reward cgc --device cuda
python scripts/attack_sneaky_only.py --model-dir outputs/goodhart_main --data-dir data/gsm8k --output-dir outputs/attacks_goodhart --reward goodhart --device cuda
```

## Expected Results

### 1. Performance Metrics

- **Verifier Accuracy**: Should reach 85-90% on test set
- **Verifier AUC**: Should reach 0.90-0.95
- **Prover Accuracy**: Should reach 80-85% for helpful role
- **Best-of-n**: Should show improvement with larger n values

### 2. Legibility Metrics

- **Human Accuracy**: Should improve over training rounds
- **Response Time**: Should decrease over training rounds
- **Confidence**: Should increase over training rounds

### 3. Attack Robustness

- **SRC**: Should be most robust to attacks
- **CGC**: Should be moderately robust
- **Goodhart**: Should be least robust (as expected)

## Troubleshooting

### 1. Common Issues

**CUDA Out of Memory**:
- Reduce batch size in config
- Use gradient accumulation
- Enable quantization

**Model Loading Issues**:
- Check HuggingFace token
- Verify model names
- Clear cache: `rm -rf .cache/`

**Data Loading Issues**:
- Check data directory structure
- Verify JSONL format
- Check file permissions

### 2. Performance Optimization

**For Large Models**:
- Use PEFT/LoRA
- Enable quantization
- Use gradient checkpointing

**For Faster Training**:
- Increase batch size
- Use mixed precision
- Enable compilation

### 3. Debugging

**Enable Debug Logging**:
```bash
export PYTHONPATH=$PWD
python scripts/run_round.py --config configs/default.yaml --verbose
```

**Check Model Outputs**:
```python
from pvg_core.parsing import extract_final_answer, is_correct
from pvg_core.sampling import ModelSampler

# Test parsing
answer = extract_final_answer("The answer is 42. #### 42")
print(f"Extracted: {answer}")

# Test sampling
sampler = ModelSampler("meta-llama/Meta-Llama-3-8B-Instruct")
responses = sampler.generate_batch(["What is 2+2?"])
print(f"Generated: {responses[0]}")
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{kirchner2024prover,
  title={Prover-Verifier Games Improve Legibility of LLM Outputs},
  author={Kirchner, Jan Hendrik and Chen, Yining and Edwards, Harri and Leike, Jan and McAleese, Nat and Burda, Yuri},
  journal={arXiv preprint arXiv:2407.13692},
  year={2024}
}
```
