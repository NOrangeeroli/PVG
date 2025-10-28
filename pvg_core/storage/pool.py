"""
Mixture-of-past-provers sample pool for training data management.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SampleMetadata:
    """Metadata for a training sample."""
    round_number: int
    role: str  # "helpful" or "sneaky"
    model_name: str
    generation_config: Dict[str, Any]
    timestamp: str
    is_correct: bool
    verifier_score: float
    reward: float

@dataclass
class TrainingSample:
    """A training sample with problem, solution, and metadata."""
    problem: str
    solution: str
    ground_truth: str
    metadata: SampleMetadata

class SamplePool:
    """Pool for managing mixture of past provers samples."""
    
    def __init__(
        self,
        pool_dir: str,
        max_samples: int = 10000,
        balance_ratio: float = 0.5
    ):
        """
        Initialize the sample pool.
        
        Args:
            pool_dir: Directory to store the pool
            max_samples: Maximum number of samples to keep
            balance_ratio: Ratio of helpful to sneaky samples
        """
        self.pool_dir = pool_dir
        self.max_samples = max_samples
        self.balance_ratio = balance_ratio
        
        # Create pool directory
        os.makedirs(pool_dir, exist_ok=True)
        
        # Load existing samples
        self.samples = self._load_samples()
        
        logger.info(f"Initialized sample pool with {len(self.samples)} samples")
    
    def _load_samples(self) -> List[TrainingSample]:
        """Load existing samples from disk."""
        samples_file = os.path.join(self.pool_dir, "samples.jsonl")
        if not os.path.exists(samples_file):
            return []
        
        samples = []
        with open(samples_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    sample = TrainingSample(
                        problem=data['problem'],
                        solution=data['solution'],
                        ground_truth=data['ground_truth'],
                        metadata=SampleMetadata(**data['metadata'])
                    )
                    samples.append(sample)
        
        return samples
    
    def _save_samples(self):
        """Save samples to disk."""
        samples_file = os.path.join(self.pool_dir, "samples.jsonl")
        with open(samples_file, 'w') as f:
            for sample in self.samples:
                data = {
                    'problem': sample.problem,
                    'solution': sample.solution,
                    'ground_truth': sample.ground_truth,
                    'metadata': asdict(sample.metadata)
                }
                f.write(json.dumps(data) + '\n')
    
    def add_samples(
        self,
        problems: List[str],
        solutions: List[str],
        ground_truths: List[str],
        roles: List[str],
        model_name: str,
        generation_config: Dict[str, Any],
        verifier_scores: List[float],
        rewards: List[float],
        round_number: int
    ):
        """
        Add new samples to the pool.
        
        Args:
            problems: List of math problems
            solutions: List of generated solutions
            ground_truths: List of ground truth answers
            roles: List of roles
            model_name: Name of the model that generated solutions
            generation_config: Configuration used for generation
            verifier_scores: List of verifier scores
            rewards: List of reward values
            round_number: Current training round
        """
        from ..parsing import extract_final_answer, is_correct
        
        new_samples = []
        timestamp = datetime.now().isoformat()
        
        for i, (problem, solution, ground_truth, role, verifier_score, reward) in enumerate(
            zip(problems, solutions, ground_truths, roles, verifier_scores, rewards)
        ):
            # Check correctness
            predicted_answer = extract_final_answer(solution)
            is_correct = is_correct(predicted_answer, ground_truth)
            
            # Create metadata
            metadata = SampleMetadata(
                round_number=round_number,
                role=role,
                model_name=model_name,
                generation_config=generation_config,
                timestamp=timestamp,
                is_correct=is_correct,
                verifier_score=verifier_score,
                reward=reward
            )
            
            # Create sample
            sample = TrainingSample(
                problem=problem,
                solution=solution,
                ground_truth=ground_truth,
                metadata=metadata
            )
            
            new_samples.append(sample)
        
        # Add to pool
        self.samples.extend(new_samples)
        
        # Trim pool if necessary
        if len(self.samples) > self.max_samples:
            self._trim_pool()
        
        # Save to disk
        self._save_samples()
        
        logger.info(f"Added {len(new_samples)} samples to pool (total: {len(self.samples)})")
    
    def _trim_pool(self):
        """Trim the pool to maintain size and balance."""
        # Sort by timestamp (keep newer samples)
        self.samples.sort(key=lambda x: x.metadata.timestamp, reverse=True)
        
        # Keep only the most recent samples
        self.samples = self.samples[:self.max_samples]
        
        # Balance helpful vs sneaky samples
        self._balance_samples()
    
    def _balance_samples(self):
        """Balance the ratio of helpful vs sneaky samples."""
        helpful_samples = [s for s in self.samples if s.metadata.role == "helpful"]
        sneaky_samples = [s for s in self.samples if s.metadata.role == "sneaky"]
        
        # Calculate target counts
        total_samples = len(self.samples)
        target_helpful = int(total_samples * self.balance_ratio)
        target_sneaky = total_samples - target_helpful
        
        # Trim if necessary
        if len(helpful_samples) > target_helpful:
            helpful_samples = helpful_samples[:target_helpful]
        
        if len(sneaky_samples) > target_sneaky:
            sneaky_samples = sneaky_samples[:target_sneaky]
        
        # Combine and shuffle
        self.samples = helpful_samples + sneaky_samples
        np.random.shuffle(self.samples)
    
    def get_training_batch(
        self,
        batch_size: int,
        balance_roles: bool = True
    ) -> List[TrainingSample]:
        """
        Get a batch of samples for training.
        
        Args:
            batch_size: Size of the batch
            balance_roles: Whether to balance helpful vs sneaky samples
            
        Returns:
            List of training samples
        """
        if balance_roles:
            # Get balanced batch
            helpful_samples = [s for s in self.samples if s.metadata.role == "helpful"]
            sneaky_samples = [s for s in self.samples if s.metadata.role == "sneaky"]
            
            # Calculate split
            half_batch = batch_size // 2
            helpful_batch = helpful_samples[:half_batch] if len(helpful_samples) >= half_batch else helpful_samples
            sneaky_batch = sneaky_samples[:half_batch] if len(sneaky_samples) >= half_batch else sneaky_samples
            
            # Combine and shuffle
            batch = helpful_batch + sneaky_batch
            np.random.shuffle(batch)
        else:
            # Get random batch
            batch = np.random.choice(self.samples, size=min(batch_size, len(self.samples)), replace=False)
        
        return batch
    
    def get_samples_by_round(self, round_number: int) -> List[TrainingSample]:
        """Get all samples from a specific round."""
        return [s for s in self.samples if s.metadata.round_number == round_number]
    
    def get_samples_by_role(self, role: str) -> List[TrainingSample]:
        """Get all samples with a specific role."""
        return [s for s in self.samples if s.metadata.role == role]
    
    def get_samples_by_model(self, model_name: str) -> List[TrainingSample]:
        """Get all samples generated by a specific model."""
        return [s for s in self.samples if s.metadata.model_name == model_name]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the sample pool."""
        if not self.samples:
            return {
                'total_samples': 0,
                'rounds': [],
                'roles': {},
                'models': {},
                'accuracy': {},
                'reward_stats': {}
            }
        
        # Basic counts
        total_samples = len(self.samples)
        rounds = sorted(set(s.metadata.round_number for s in self.samples))
        
        # By role
        role_counts = {}
        for role in ['helpful', 'sneaky']:
            role_samples = [s for s in self.samples if s.metadata.role == role]
            role_counts[role] = len(role_samples)
        
        # By model
        model_counts = {}
        for sample in self.samples:
            model = sample.metadata.model_name
            model_counts[model] = model_counts.get(model, 0) + 1
        
        # Accuracy by role
        accuracy_by_role = {}
        for role in ['helpful', 'sneaky']:
            role_samples = [s for s in self.samples if s.metadata.role == role]
            if role_samples:
                correct_count = sum(1 for s in role_samples if s.metadata.is_correct)
                accuracy_by_role[role] = correct_count / len(role_samples)
        
        # Reward statistics
        rewards = [s.metadata.reward for s in self.samples]
        reward_stats = {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards))
        }
        
        return {
            'total_samples': total_samples,
            'rounds': rounds,
            'roles': role_counts,
            'models': model_counts,
            'accuracy': accuracy_by_role,
            'reward_stats': reward_stats
        }
    
    def clear_pool(self):
        """Clear all samples from the pool."""
        self.samples = []
        self._save_samples()
        logger.info("Cleared sample pool")
    
    def export_samples(self, output_file: str, format: str = "jsonl"):
        """Export samples to a file."""
        if format == "jsonl":
            with open(output_file, 'w') as f:
                for sample in self.samples:
                    data = {
                        'problem': sample.problem,
                        'solution': sample.solution,
                        'ground_truth': sample.ground_truth,
                        'metadata': asdict(sample.metadata)
                    }
                    f.write(json.dumps(data) + '\n')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(self.samples)} samples to {output_file}")
    
    def import_samples(self, input_file: str, format: str = "jsonl"):
        """Import samples from a file."""
        if format == "jsonl":
            with open(input_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        sample = TrainingSample(
                            problem=data['problem'],
                            solution=data['solution'],
                            ground_truth=data['ground_truth'],
                            metadata=SampleMetadata(**data['metadata'])
                        )
                        self.samples.append(sample)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save to disk
        self._save_samples()
        logger.info(f"Imported samples from {input_file}")
