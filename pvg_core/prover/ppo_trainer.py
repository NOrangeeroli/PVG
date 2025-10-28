"""
PPO trainer for training prover models with reward functions.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from trl import PPOTrainer, PPOConfig
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from tqdm import tqdm
import json
import os

from .policy_loader import ProverPolicyLoader, ReferenceModelLoader
from ..rewards import RewardCalculator, RewardConfig, calculate_kl_penalty, calculate_entropy_bonus
from ..parsing import extract_final_answer, is_correct

logger = logging.getLogger(__name__)

class ProverPPOTrainer:
    """PPO trainer for prover models."""
    
    def __init__(
        self,
        policy_loader: ProverPolicyLoader,
        reference_loader: ReferenceModelLoader,
        reward_config: RewardConfig,
        ppo_config: PPOConfig,
        device: str = "cuda",
        use_multi_gpu: bool = False,
        **kwargs
    ):
        """
        Initialize the PPO trainer.
        
        Args:
            policy_loader: Loader for the policy model
            reference_loader: Loader for the reference model
            reward_config: Configuration for reward functions
            ppo_config: Configuration for PPO training
            device: Device to use for training
            **kwargs: Additional arguments
        """
        self.policy_loader = policy_loader
        self.reference_loader = reference_loader
        self.reward_config = reward_config
        self.ppo_config = ppo_config
        self.device = device
        
        # Get models
        self.policy_model = policy_loader.get_model()
        self.reference_model = reference_loader.get_model()
        self.tokenizer = policy_loader.get_tokenizer()
        
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(reward_config)
        
        # Training history
        self.train_history = []
        self.use_multi_gpu = use_multi_gpu
        
        # Setup multi-GPU if enabled
        if use_multi_gpu and torch.cuda.device_count() > 1:
            logger.info("Using %d GPUs for PPO training", torch.cuda.device_count())
            self.policy_model = torch.nn.DataParallel(self.policy_model)
            self.reference_model = torch.nn.DataParallel(self.reference_model)
        
        # Move models to device
        self.policy_model.to(device)
        self.reference_model.to(device)
    
    def generate_responses(
        self,
        problems: List[str],
        roles: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of problems with roles.
        
        Args:
            problems: List of math problems
            roles: List of roles ("helpful" or "sneaky")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses with metadata
        """
        from ..prompts import format_simple_prompt
        
        # Format prompts with roles
        prompts = []
        for problem, role in zip(problems, roles):
            prompt = format_simple_prompt(problem, role)
            prompts.append(prompt)
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate responses
        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        # Decode responses
        responses = []
        for i, output in enumerate(outputs):
            # Remove input tokens
            input_length = inputs['input_ids'][i].shape[0]
            generated_tokens = output[input_length:]
            
            # Decode
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            responses.append({
                'problem': problems[i],
                'role': roles[i],
                'solution': response,
                'prompt': prompts[i],
                'input_length': input_length,
                'generation_length': len(generated_tokens)
            })
        
        return responses
    
    def calculate_rewards(
        self,
        responses: List[Dict[str, Any]],
        verifier_scores: List[float],
        ground_truth_answers: List[str]
    ) -> List[float]:
        """
        Calculate rewards for generated responses.
        
        Args:
            responses: List of generated responses
            verifier_scores: List of verifier scores
            ground_truth_answers: List of ground truth answers
            
        Returns:
            List of reward values
        """
        # Prepare batch data for reward calculation
        batch_data = []
        for i, response in enumerate(responses):
            # Extract final answer
            predicted_answer = extract_final_answer(response['solution'])
            
            # Check correctness
            is_correct = is_correct(predicted_answer, ground_truth_answers[i])
            
            batch_data.append({
                'verifier_score': verifier_scores[i],
                'is_correct': is_correct,
                'role': response['role']
            })
        
        # Calculate rewards
        rewards = self.reward_calculator.calculate_rewards(batch_data)
        
        return rewards
    
    def calculate_kl_penalty(
        self,
        responses: List[Dict[str, Any]],
        beta: float = 0.02
    ) -> torch.Tensor:
        """
        Calculate KL penalty between policy and reference model.
        
        Args:
            responses: List of generated responses
            beta: KL penalty coefficient
            
        Returns:
            KL penalty tensor
        """
        kl_penalties = []
        
        for response in responses:
            # Tokenize the full prompt + response
            full_text = response['prompt'] + response['solution']
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Get log probabilities from both models
            with torch.no_grad():
                # Policy model
                policy_outputs = self.policy_model(**inputs)
                policy_logits = policy_outputs.logits
                policy_log_probs = F.log_softmax(policy_logits, dim=-1)
                
                # Reference model
                ref_outputs = self.reference_model(**inputs)
                ref_logits = ref_outputs.logits
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                
                # Calculate KL divergence
                kl_div = F.kl_div(
                    policy_log_probs,
                    ref_log_probs,
                    reduction='batchmean'
                )
                
                kl_penalties.append(kl_div.item())
        
        return torch.tensor(kl_penalties) * beta
    
    def train_step(
        self,
        problems: List[str],
        roles: List[str],
        verifier_scores: List[float],
        ground_truth_answers: List[str],
        kl_beta: float = 0.02,
        entropy_coef: float = 0.01,
        **kwargs
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            problems: List of math problems
            roles: List of roles
            verifier_scores: List of verifier scores
            ground_truth_answers: List of ground truth answers
            kl_beta: KL penalty coefficient
            entropy_coef: Entropy bonus coefficient
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        # Generate responses
        responses = self.generate_responses(problems, roles, **kwargs)
        
        # Calculate rewards
        rewards = self.calculate_rewards(responses, verifier_scores, ground_truth_answers)
        
        # Calculate KL penalty
        kl_penalty = self.calculate_kl_penalty(responses, kl_beta)
        
        # Calculate combined rewards
        combined_rewards = torch.tensor(rewards) - kl_penalty
        
        # Calculate metrics
        metrics = {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'mean_kl_penalty': float(kl_penalty.mean()),
            'mean_combined_reward': float(combined_rewards.mean()),
            'num_responses': len(responses),
            'accuracy': float(np.mean([
                is_correct(extract_final_answer(r['solution']), gt)
                for r, gt in zip(responses, ground_truth_answers)
            ]))
        }
        
        # Analyze reward distribution
        reward_stats = self.reward_calculator.analyze_reward_distribution(
            rewards, roles, [is_correct(extract_final_answer(r['solution']), gt) 
                           for r, gt in zip(responses, ground_truth_answers)]
        )
        metrics.update(reward_stats)
        
        return metrics
    
    def train(
        self,
        train_data: List[Dict[str, Any]],
        num_epochs: int = 3,
        batch_size: int = 8,
        save_every: int = 1,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Train the prover model with PPO.
        
        Args:
            train_data: List of training examples with keys: problem, role, verifier_score, ground_truth
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            save_every: Save model every N epochs
            save_dir: Directory to save models
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting PPO training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Shuffle data
            np.random.shuffle(train_data)
            
            # Process in batches with progress bar
            epoch_metrics = []
            num_batches = (len(train_data) + batch_size - 1) // batch_size
            
            batch_pbar = tqdm(
                range(0, len(train_data), batch_size),
                total=num_batches,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                unit="batch"
            )
            
            for i in batch_pbar:
                batch = train_data[i:i + batch_size]
                
                # Extract batch data
                problems = [item['problem'] for item in batch]
                roles = [item['role'] for item in batch]
                verifier_scores = [item['verifier_score'] for item in batch]
                ground_truths = [item['ground_truth'] for item in batch]
                
                # Training step
                step_metrics = self.train_step(
                    problems, roles, verifier_scores, ground_truths, **kwargs
                )
                
                epoch_metrics.append(step_metrics)
                
                # Update progress bar
                batch_pbar.set_postfix({
                    'loss': f"{step_metrics.get('loss', 0):.4f}",
                    'reward': f"{step_metrics.get('reward', 0):.4f}",
                    'kl': f"{step_metrics.get('kl_divergence', 0):.4f}"
                })
            
            batch_pbar.close()
            
            # Calculate epoch metrics
            epoch_summary = {}
            for key in epoch_metrics[0].keys():
                values = [m[key] for m in epoch_metrics]
                epoch_summary[f'mean_{key}'] = float(np.mean(values))
                epoch_summary[f'std_{key}'] = float(np.std(values))
            
            self.train_history.append(epoch_summary)
            logger.info(f"Epoch {epoch + 1} summary: {epoch_summary}")
            
            # Save model
            if save_dir and (epoch + 1) % save_every == 0:
                self.save_model(save_dir, epoch + 1)
        
        return {'train': self.train_history}
    
    def save_model(self, save_dir: str, epoch: int):
        """Save the model and training history."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save policy model
        policy_dir = f"{save_dir}/policy_epoch_{epoch}"
        self.policy_loader.save_pretrained(policy_dir)
        
        # Save reference model
        ref_dir = f"{save_dir}/reference_epoch_{epoch}"
        self.reference_loader.save_pretrained(ref_dir)
        
        # Save training history
        history = {
            'train': self.train_history,
            'epoch': epoch,
            'reward_config': self.reward_config.__dict__,
            'ppo_config': self.ppo_config.__dict__
        }
        
        with open(f"{save_dir}/training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Model saved to {save_dir}")
    
    def load_model(self, model_dir: str):
        """Load a pretrained model."""
        # Load policy model
        policy_dir = f"{model_dir}/policy_epoch_latest"
        self.policy_loader.load_pretrained(policy_dir)
        
        # Load reference model
        ref_dir = f"{model_dir}/reference_epoch_latest"
        self.reference_loader.load_pretrained(ref_dir)
        
        # Load training history
        history_file = f"{model_dir}/training_history.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
                self.train_history = history.get('train', [])
        
        logger.info(f"Model loaded from {model_dir}")
