#!/usr/bin/env python3
"""
Run one complete training round: Verifier SL -> Prover PPO.
"""

import argparse
import json
import os
import logging
from typing import Dict, Any, List, Tuple
import torch
import numpy as np
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

# Import our modules
from pvg_core.verifier import VerifierModel, VerifierTokenizer, VerifierTrainer, VerifierDataset
from pvg_core.prover import ProverPolicyLoader, ReferenceModelLoader, ProverPPOTrainer
from pvg_core.storage import SamplePool, RunManager
from pvg_core.rewards import RewardConfig
from pvg_core.parsing import extract_final_answer, is_correct
from pvg_core.sampling import ModelSampler

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    if os.path.exists(config_path):
        return OmegaConf.load(config_path)
    else:
        # Return default config
        return {
            'prover': {
                'model': 'meta-llama/Meta-Llama-3-8B-Instruct',
                'use_peft': True,
                'peft_config': {
                    'r': 16,
                    'lora_alpha': 32,
                    'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                    'lora_dropout': 0.1
                }
            },
            'verifier': {
                'model': 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-503b',
                'num_classes': 2,
                'dropout_rate': 0.1
            },
            'rl': {
                'reward': 'src',
                'kl_beta': 0.02,
                'entropy_coef': 0.01,
                'batch_size': 8,
                'num_epochs': 3
            },
            'data': {
                'train_size': 1000,
                'val_size': 200,
                'test_size': 200
            },
            'logging': {
                'save_every': 1,
                'eval_every': 1
            }
        }

def load_data(data_dir: str, split: str = 'train', max_problems: int = None) -> List[Dict[str, Any]]:
    """Load data from JSONL file with optional limit."""
    data_path = os.path.join(data_dir, f"{split}.jsonl")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                # Apply limit if specified
                if max_problems is not None and len(data) >= max_problems:
                    break
    
    logger.info(f"Loaded {len(data)} {split} samples from {data_path}" + 
                (f" (limited to {max_problems})" if max_problems else ""))
    return data

def split_training_data(
    train_problems: List[Dict[str, Any]], 
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split training data in half for prover and verifier training.
    
    Args:
        train_problems: List of training problems
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (prover_problems, verifier_problems)
    """
    import random
    random.seed(seed)
    
    # Shuffle the data
    shuffled_data = train_problems.copy()
    random.shuffle(shuffled_data)
    
    # Split in half
    split_idx = len(shuffled_data) // 2
    prover_problems = shuffled_data[:split_idx]
    verifier_problems = shuffled_data[split_idx:]
    
    logger.info("Split training data: %d problems for prover, %d problems for verifier", 
                len(prover_problems), len(verifier_problems))
    
    return prover_problems, verifier_problems

def prepare_verifier_training_data(
    problems: List[Dict[str, Any]],
    prover_loader: ProverPolicyLoader,
    config: Dict[str, Any],
    round_num: int,
    split: str = 'train',
    output_dir: str = "."
) -> List[Dict[str, Any]]:
    """
    Sample solutions from helpful/sneaky provers for verifier training.
    
    For each problem:
    - Sample k correct solutions from helpful prover
    - Sample k incorrect solutions from sneaky prover
    - Return balanced dataset with is_correct labels
    """
    from pvg_core.verifier.sample_solutions import sample_solutions_for_verifier, load_cached_solutions, cache_solutions
    
    cache_dir = config['verifier']['sampling']['cache_dir']
    cache_file = f"{cache_dir}/round_{round_num}_{split}_samples.json"
    
    # Check cache first
    if config['verifier']['sampling']['use_cached'] and os.path.exists(cache_file):
        logger.info("Loading cached verifier samples from %s", cache_file)
        cached_solutions = load_cached_solutions(cache_file)
        if cached_solutions is not None:
            return cached_solutions
    
    # Sample new solutions
    logger.info("Sampling solutions for verifier training (round %d)", round_num)
    verifier_data = sample_solutions_for_verifier(
        problems=problems,
        prover_model=prover_loader.get_model(),
        tokenizer=prover_loader.get_tokenizer(),
        k_per_role=config['verifier']['sampling']['k_per_role'],
        max_tries_per_task=config['verifier']['sampling']['max_tries_per_task'],
        round_num=round_num,
        use_fast_sampling=config['verifier']['sampling']['use_fast_sampling'],
        task_batch_size=config['verifier']['sampling']['batch_size_per_device'],
        data_parallel_size=config['verifier']['sampling'].get('data_parallel_size'),
        gpu_memory_utilization=config['verifier']['sampling'].get('gpu_memory_utilization', 0.8),
        prover_model_name=config['prover']['model'],  # Pass model name from config
        gpu_ids=config['verifier']['sampling'].get('gpu_ids'),
        avoid_conflicts=config['verifier']['sampling'].get('avoid_conflicts', False),
        output_dir=output_dir,  # Pass output directory for reports
    )
    
    # Cache for future use
    cache_solutions(verifier_data, cache_file)
    
    return verifier_data

def train_verifier(
    verifier_model: VerifierModel,
    verifier_tokenizer: VerifierTokenizer,
    train_problems: List[Dict[str, Any]],  # Changed from train_data
    val_problems: List[Dict[str, Any]],    # Changed from val_data
    prover_loader: ProverPolicyLoader,      # NEW
    config: Dict[str, Any],
    round_num: int,                         # NEW
    device: str = "cuda",
    multi_gpu: bool = False,
    output_dir: str = "outputs"             # NEW: Output directory for reports
) -> VerifierModel:
    """Train the verifier model."""
    logger.info("Training verifier...")
    
    # Sample solutions from provers for verifier training
    train_data = prepare_verifier_training_data(
        train_problems, prover_loader, config, round_num, 'train', output_dir
    )
    val_data = prepare_verifier_training_data(
        val_problems, prover_loader, config, round_num, 'val', output_dir
    )
    
    # Create datasets
    train_dataset = VerifierDataset(train_data, verifier_tokenizer)
    val_dataset = VerifierDataset(val_data, verifier_tokenizer)
    
    # Create data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['verifier']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['verifier']['batch_size'], shuffle=False)
    
    # Create trainer
    trainer = VerifierTrainer(
        verifier_model,
        verifier_tokenizer,
        device=device,
        learning_rate=2e-5,
        weight_decay=0.01,
        use_multi_gpu=multi_gpu
    )
    
    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=3,
        save_every=1,
        save_dir="outputs/verifier"
    )
    
    logger.info("Verifier training complete")
    return verifier_model

def train_prover(
    prover_loader: ProverPolicyLoader,
    reference_loader: ReferenceModelLoader,
    verifier_model: VerifierModel,
    train_data: List[Dict[str, Any]],
    config: Dict[str, Any],
    device: str = "cuda"
) -> ProverPolicyLoader:
    """Train the prover model with PPO."""
    logger.info("Training prover...")
    
    # Create reward config
    reward_config = RewardConfig(
        reward_type=config['rl']['reward'],
        verifier_score_key='verifier_score',
        correctness_key='is_correct',
        role_key='role'
    )
    
    # Create PPO config
    from trl import PPOConfig
    ppo_config = PPOConfig(
        batch_size=config['rl']['batch_size'],
        mini_batch_size=4,
        ppo_epochs=config['rl']['num_epochs'],
        learning_rate=1e-5,
        max_grad_norm=1.0
    )
    
    # Create trainer
    trainer = ProverPPOTrainer(
        prover_loader,
        reference_loader,
        reward_config,
        ppo_config,
        device=device
    )
    
    # Prepare training data
    training_data = []
    for item in train_data:
        # Generate solutions for this problem
        problems = [item['problem']]
        roles = ['helpful']  # Start with helpful role
        
        # Generate solutions
        responses = trainer.generate_responses(problems, roles)
        
        # Get verifier scores
        verifier_scores = []
        for response in responses:
            score = verifier_model.get_verifier_scores(
                [(response['problem'], response['solution'])]
            )[0]
            verifier_scores.append(score)
        
        # Calculate rewards
        rewards = trainer.calculate_rewards(
            responses, verifier_scores, [item['ground_truth']]
        )
        
        # Add to training data
        for response, score, reward in zip(responses, verifier_scores, rewards):
            training_data.append({
                'problem': response['problem'],
                'role': response['role'],
                'verifier_score': score,
                'ground_truth': item['ground_truth'],
                'reward': reward
            })
    
    # Train
    history = trainer.train(
        training_data,
        num_epochs=config['rl']['num_epochs'],
        batch_size=config['rl']['batch_size'],
        save_every=1,
        save_dir="outputs/prover"
    )
    
    logger.info("Prover training complete")
    return prover_loader

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run one training round")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--data-dir", default="data/gsm8k", help="Data directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--round", type=int, default=0, help="Training round number")
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs for training")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config first to check for GPU settings
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set GPU visibility BEFORE any GPU operations
    gpu_ids = config.get('verifier', {}).get('sampling', {}).get('gpu_ids')
    if gpu_ids is not None:
        gpu_str = ",".join(map(str, gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        logging.info(f"Set CUDA_VISIBLE_DEVICES={gpu_str} before any GPU operations")
    
    # Re-import torch after setting CUDA_VISIBLE_DEVICES
    import torch
    torch.cuda.empty_cache()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data - these are now just problems, not problem-solution pairs
    # Apply testing limits if specified
    max_problems = config.get('testing', {}).get('max_problems')
    max_verifier_problems = config.get('testing', {}).get('max_verifier_problems')
    max_prover_problems = config.get('testing', {}).get('max_prover_problems')
    quick_test = config.get('testing', {}).get('quick_test', False)
    
    # Set quick test limits if enabled
    if quick_test:
        max_problems = max_problems or 20
        max_verifier_problems = max_verifier_problems or 10
        max_prover_problems = max_prover_problems or 10
    
    train_problems = load_data(args.data_dir, 'train', max_problems)
    val_problems = load_data(args.data_dir, 'val', max_problems)
    
    # Split training data in half for prover and verifier (they never see same prompts)
    prover_problems, verifier_problems = split_training_data(train_problems, seed=args.seed)
    
    # Apply specific limits to prover and verifier problems
    if max_prover_problems is not None:
        prover_problems = prover_problems[:max_prover_problems]
        logger.info(f"Limited prover problems to {len(prover_problems)}")
    
    if max_verifier_problems is not None:
        verifier_problems = verifier_problems[:max_verifier_problems]
        logger.info(f"Limited verifier problems to {len(verifier_problems)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize models
    logger.info("Initializing models...")
    
    # Initialize prover FIRST (needed for verifier training)
    prover_loader = ProverPolicyLoader(
        model_name=config['prover']['model'],
        use_peft=config['prover']['use_peft'],
        peft_config=config['prover']['peft_config']
    )
    
    # Reference model
    reference_loader = ReferenceModelLoader(
        model_name=config['prover']['model']
    )
    
    # Verifier
    verifier_model = VerifierModel(
        base_model_name=config['verifier']['model'],
        num_classes=config['verifier']['num_classes'],
        dropout_rate=config['verifier']['dropout_rate']
    )
    verifier_tokenizer = VerifierTokenizer(
        tokenizer_name=config['verifier']['model'],
        max_length=512
    )
    
    # Use few-shot prompting for Round 0 (base prover)
    # (For now, use the initialized prover as-is)
    
    # Overall training progress
    training_steps = [
        "Initialize models",
        "Train verifier", 
        "Train prover",
        "Save models"
    ]
    
    overall_pbar = tqdm(
        training_steps,
        desc="Training round",
        unit="step"
    )
    
    # Initialize models
    overall_pbar.set_description("Initializing models")
    overall_pbar.update(1)
    
    # Train verifier with sampled solutions (using verifier's half of training data)
    overall_pbar.set_description("Training verifier")
    verifier_model = train_verifier(
        verifier_model, 
        verifier_tokenizer, 
        verifier_problems,  # Verifier's half of training data
        val_problems,      # Validation data
        prover_loader,     # Prover for sampling
        config,
        round_num=args.round,
        device=args.device,
        multi_gpu=args.multi_gpu,
        output_dir=args.output_dir
    )
    overall_pbar.update(1)
    
    # Train prover (using prover's half of training data)
    overall_pbar.set_description("Training prover")
    prover_loader = train_prover(
        prover_loader, reference_loader, verifier_model, prover_problems, config, args.device
    )
    overall_pbar.update(1)
    
    # Save models
    overall_pbar.set_description("Saving models")
    verifier_model.save_pretrained(f"{args.output_dir}/verifier")
    prover_loader.save_pretrained(f"{args.output_dir}/prover")
    reference_loader.save_pretrained(f"{args.output_dir}/reference")
    overall_pbar.update(1)
    overall_pbar.close()
    
    logger.info("Training round complete!")

if __name__ == "__main__":
    main()
