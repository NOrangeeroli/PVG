# PVG-Legibility Documentation

This directory contains comprehensive documentation for the PVG-Legibility project.

## Documentation Structure

### 1. Core Documentation

- **README.md** (main project): Main project documentation
- **REPRODUCIBILITY.md**: Detailed reproduction instructions
- **HUMAN_STUDY_GUIDE.md**: Human evaluation study guide

### 2. API Documentation

- **API_REFERENCE.md**: Complete API reference
- **MODULE_GUIDE.md**: Module-by-module guide
- **EXAMPLES.md**: Code examples and tutorials

### 3. Advanced Topics

- **CUSTOMIZATION.md**: Customizing models and training
- **PERFORMANCE.md**: Performance optimization guide
- **TROUBLESHOOTING.md**: Common issues and solutions

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-org/pvg-legibility.git
cd pvg-legibility

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Basic Usage

```bash
# Prepare data
python scripts/prepare_gsm8k.py --output-dir data/gsm8k

# Train one round
python scripts/run_round.py --config configs/default.yaml --data-dir data/gsm8k --output-dir outputs/round_001

# Evaluate
python scripts/eval_all.py --model-dir outputs/round_001 --data-dir data/gsm8k --output-dir outputs/eval_001
```

### 3. Advanced Usage

```bash
# Multi-round training
for round in {1..5}; do
  python scripts/run_round.py \
    --config configs/default.yaml \
    --data-dir data/gsm8k \
    --output-dir outputs/round_$(printf "%03d" $round)
done

# Attack evaluation
python scripts/attack_sneaky_only.py \
  --model-dir outputs/round_001 \
  --data-dir data/gsm8k \
  --output-dir outputs/attacks_001 \
  --reward goodhart
```

## Configuration

### 1. Default Configuration

The default configuration (`configs/default.yaml`) provides a good starting point for most experiments.

### 2. Experiment Configurations

- **SRC Reward**: `configs/experiments.yaml:src_experiment`
- **CGC Reward**: `configs/experiments.yaml:cgc_experiment`
- **Goodharting**: `configs/experiments.yaml:goodhart_experiment`

### 3. Custom Configuration

Create your own configuration by extending the default:

```yaml
# configs/my_experiment.yaml
defaults:
  - default
  - _self_

prover:
  model: "meta-llama/Meta-Llama-3-70B-Instruct"
  use_quantization: true

rl:
  reward: "src"
  kl_beta: 0.01
  batch_size: 16
```

## Modules

### 1. Core Modules

- **prompts.py**: Prompt templates and formatting
- **parsing.py**: Answer extraction and validation
- **sampling.py**: Model sampling utilities
- **rewards.py**: Reward function implementations

### 2. Verifier Module

- **model.py**: Verifier model implementation
- **train.py**: Verifier training loop

### 3. Prover Module

- **policy_loader.py**: Prover model loading
- **ppo_trainer.py**: PPO training implementation

### 4. Storage Module

- **pool.py**: Sample pool management
- **runs.py**: Run and checkpoint management

### 5. Evaluation Module

- **metrics.py**: Evaluation metrics
- **best_of_n.py**: Best-of-n evaluation
- **attacks.py**: Adversarial attack evaluation
- **human_proxy.py**: Human proxy evaluation

## Scripts

### 1. Data Preparation

```bash
# Download and prepare GSM8K dataset
python scripts/prepare_gsm8k.py --output-dir data/gsm8k --subset 1000
```

### 2. Training

```bash
# Single round training
python scripts/run_round.py --config configs/default.yaml --data-dir data/gsm8k --output-dir outputs/round_001

# Multi-round training
python scripts/run_multi_round.py --config configs/default.yaml --data-dir data/gsm8k --output-dir outputs/multi_round --num-rounds 5
```

### 3. Evaluation

```bash
# Comprehensive evaluation
python scripts/eval_all.py --model-dir outputs/round_001 --data-dir data/gsm8k --output-dir outputs/eval_001

# Attack evaluation
python scripts/attack_sneaky_only.py --model-dir outputs/round_001 --data-dir data/gsm8k --output-dir outputs/attacks_001 --reward goodhart
```

## Examples

### 1. Basic Training Loop

```python
from pvg_core.verifier import VerifierModel, VerifierTokenizer, VerifierTrainer
from pvg_core.prover import ProverPolicyLoader, ProverPPOTrainer
from pvg_core.rewards import RewardConfig

# Initialize models
verifier_model = VerifierModel("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-503b")
verifier_tokenizer = VerifierTokenizer("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-503b")
prover_loader = ProverPolicyLoader("meta-llama/Meta-Llama-3-8B-Instruct")

# Train verifier
trainer = VerifierTrainer(verifier_model, verifier_tokenizer)
trainer.train(train_loader, val_loader, num_epochs=3)

# Train prover
reward_config = RewardConfig(reward_type="src")
ppo_trainer = ProverPPOTrainer(prover_loader, reference_loader, reward_config, ppo_config)
ppo_trainer.train(training_data, num_epochs=3)
```

### 2. Custom Reward Function

```python
from pvg_core.rewards import RewardCalculator, RewardConfig

# Create custom reward config
config = RewardConfig(
    reward_type="custom",
    verifier_score_key="verifier_score",
    correctness_key="is_correct",
    role_key="role"
)

# Implement custom reward function
class CustomRewardCalculator(RewardCalculator):
    def _calculate_custom_rewards(self, batch_data):
        # Your custom reward logic here
        rewards = []
        for data in batch_data:
            # Custom reward calculation
            reward = self._custom_reward_logic(data)
            rewards.append(reward)
        return rewards

# Use custom calculator
calculator = CustomRewardCalculator(config)
rewards = calculator.calculate_rewards(batch_data)
```

### 3. Evaluation Pipeline

```python
from pvg_core.eval import BestOfNEvaluator, AttackEvaluator, HumanProxyEvaluator

# Best-of-n evaluation
best_of_n_evaluator = BestOfNEvaluator(verifier_model, verifier_tokenizer)
best_of_n_results = best_of_n_evaluator.evaluate_best_of_n(balanced_sets, n_values=[1, 5, 10, 20])

# Attack evaluation
attack_evaluator = AttackEvaluator(verifier_model, verifier_tokenizer)
robustness_metrics = attack_evaluator.evaluate_verifier_robustness(attack_results)

# Human proxy evaluation
human_evaluator = HumanProxyEvaluator(verifier_model, verifier_tokenizer, time_limit=45.0)
human_results = human_evaluator.evaluate_batch(problems, solutions, ground_truths)
```

## Troubleshooting

### 1. Common Issues

**CUDA Out of Memory**:
- Reduce batch size
- Use gradient accumulation
- Enable quantization
- Use smaller models

**Model Loading Issues**:
- Check HuggingFace token
- Verify model names
- Clear cache
- Check disk space

**Training Instability**:
- Adjust learning rates
- Check gradient norms
- Monitor KL divergence
- Use warmup

### 2. Performance Optimization

**For Large Models**:
- Use PEFT/LoRA
- Enable quantization
- Use gradient checkpointing
- Optimize memory usage

**For Faster Training**:
- Increase batch size
- Use mixed precision
- Enable compilation
- Use multiple GPUs

### 3. Debugging

**Enable Debug Logging**:
```bash
export PYTHONPATH=$PWD
python scripts/run_round.py --config configs/default.yaml --verbose
```

**Check Model Outputs**:
```python
# Test parsing
from pvg_core.parsing import extract_final_answer, is_correct
answer = extract_final_answer("The answer is 42. #### 42")
print(f"Extracted: {answer}")

# Test sampling
from pvg_core.sampling import ModelSampler
sampler = ModelSampler("meta-llama/Meta-Llama-3-8B-Instruct")
responses = sampler.generate_batch(["What is 2+2?"])
print(f"Generated: {responses[0]}")
```

## Contributing

### 1. Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
black pvg_core/
flake8 pvg_core/
mypy pvg_core/
```

### 2. Code Style

- Follow PEP 8
- Use type hints
- Write docstrings
- Add tests
- Update documentation

### 3. Pull Request Process

1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Update documentation
6. Submit pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

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

## Support

For questions and support:

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@example.com
- **Documentation**: This directory

## Changelog

### Version 0.1.0

- Initial release
- Core functionality implementation
- Basic training and evaluation
- Documentation and examples
