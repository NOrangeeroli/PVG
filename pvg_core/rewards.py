"""
Reward functions for training provers with different objectives.

Implements SRC, CGC, and Goodharting reward functions as described in the paper.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    reward_type: str = "src"  # "src", "cgc", "goodhart"
    verifier_score_key: str = "verifier_score"
    correctness_key: str = "is_correct"
    role_key: str = "role"
    default_penalty: float = -2.0
    batch_normalize: bool = True

class RewardCalculator:
    """Calculator for different reward functions."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
    
    def calculate_rewards(
        self,
        batch_data: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Calculate rewards for a batch of data.
        
        Args:
            batch_data: List of dictionaries with keys:
                - verifier_score: float (0-1)
                - is_correct: bool
                - role: str ("helpful" or "sneaky")
                
        Returns:
            List of reward values
        """
        if self.config.reward_type == "src":
            return self._calculate_src_rewards(batch_data)
        elif self.config.reward_type == "cgc":
            return self._calculate_cgc_rewards(batch_data)
        elif self.config.reward_type == "goodhart":
            return self._calculate_goodhart_rewards(batch_data)
        else:
            raise ValueError(f"Unknown reward type: {self.config.reward_type}")
    
    def _calculate_src_rewards(self, batch_data: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate Signed Relative Convincingness (SRC) rewards.
        
        R_SRC(z_i | x, role) = (2c - 1) * (2h - 1) * V'(x, z_i) if V'(x, z_i) > 0
                              -1 otherwise
        """
        # Extract scores and normalize within batch
        verifier_scores = [d[self.config.verifier_score_key] for d in batch_data]
        
        if self.config.batch_normalize:
            # Batch normalization: mean 0, std 1
            scores_array = np.array(verifier_scores)
            if len(scores_array) > 1:
                normalized_scores = (scores_array - scores_array.mean()) / (scores_array.std() + 1e-8)
            else:
                normalized_scores = scores_array
        else:
            normalized_scores = np.array(verifier_scores)
        
        rewards = []
        for i, data in enumerate(batch_data):
            verifier_score = data[self.config.verifier_score_key]
            is_correct = data[self.config.correctness_key]
            role = data[self.config.role_key]
            
            # Convert to binary indicators
            c = 1.0 if is_correct else 0.0  # correctness
            h = 1.0 if role == "helpful" else 0.0  # helpful role
            
            # Calculate SRC reward
            if verifier_score > 0:
                reward = (2 * c - 1) * (2 * h - 1) * normalized_scores[i]
            else:
                reward = -1.0
            
            rewards.append(reward)
        
        return rewards
    
    def _calculate_cgc_rewards(self, batch_data: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate Correctness-Gated Convincingness (CGC) rewards.
        
        R_CGC(z | x, role) = V(x, z) if h == c
                            V_0 otherwise
        """
        rewards = []
        for data in batch_data:
            verifier_score = data[self.config.verifier_score_key]
            is_correct = data[self.config.correctness_key]
            role = data[self.config.role_key]
            
            # Check if role and correctness align
            h = 1.0 if role == "helpful" else 0.0
            c = 1.0 if is_correct else 0.0
            
            if h == c:  # Role and correctness align
                reward = verifier_score
            else:  # Misaligned
                reward = self.config.default_penalty
            
            rewards.append(reward)
        
        return rewards
    
    def _calculate_goodhart_rewards(self, batch_data: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate Goodharting rewards (no role conditioning).
        
        R_goodhart(z | x) = V(x, z)
        """
        return [d[self.config.verifier_score_key] for d in batch_data]

def calculate_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    beta: float = 0.02
) -> torch.Tensor:
    """
    Calculate KL penalty between current and reference policy.
    
    Args:
        log_probs: Current policy log probabilities
        ref_log_probs: Reference policy log probabilities
        beta: KL penalty coefficient
        
    Returns:
        KL penalty tensor
    """
    kl_div = torch.sum(log_probs - ref_log_probs, dim=-1)
    return beta * kl_div

def calculate_entropy_bonus(
    log_probs: torch.Tensor,
    entropy_coef: float = 0.01
) -> torch.Tensor:
    """
    Calculate entropy bonus to encourage exploration.
    
    Args:
        log_probs: Policy log probabilities
        entropy_coef: Entropy coefficient
        
    Returns:
        Entropy bonus tensor
    """
    entropy = -torch.sum(log_probs * torch.exp(log_probs), dim=-1)
    return entropy_coef * entropy

def calculate_combined_reward(
    base_rewards: List[float],
    kl_penalty: Optional[torch.Tensor] = None,
    entropy_bonus: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Combine base rewards with KL penalty and entropy bonus.
    
    Args:
        base_rewards: List of base reward values
        kl_penalty: Optional KL penalty tensor
        entropy_bonus: Optional entropy bonus tensor
        
    Returns:
        Combined reward tensor
    """
    rewards = torch.tensor(base_rewards, dtype=torch.float32)
    
    if kl_penalty is not None:
        rewards = rewards - kl_penalty
    
    if entropy_bonus is not None:
        rewards = rewards + entropy_bonus
    
    return rewards

def analyze_reward_distribution(
    rewards: List[float],
    roles: List[str],
    correctness: List[bool]
) -> Dict[str, Any]:
    """
    Analyze the distribution of rewards across different conditions.
    
    Args:
        rewards: List of reward values
        roles: List of roles
        correctness: List of correctness labels
        
    Returns:
        Dictionary with reward statistics
    """
    rewards = np.array(rewards)
    roles = np.array(roles)
    correctness = np.array(correctness)
    
    stats = {
        'overall': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards))
        },
        'by_role': {},
        'by_correctness': {},
        'by_role_and_correctness': {}
    }
    
    # By role
    for role in ['helpful', 'sneaky']:
        mask = roles == role
        if np.any(mask):
            stats['by_role'][role] = {
                'mean': float(np.mean(rewards[mask])),
                'std': float(np.std(rewards[mask])),
                'count': int(np.sum(mask))
            }
    
    # By correctness
    for correct in [True, False]:
        mask = correctness == correct
        if np.any(mask):
            stats['by_correctness'][str(correct)] = {
                'mean': float(np.mean(rewards[mask])),
                'std': float(np.std(rewards[mask])),
                'count': int(np.sum(mask))
            }
    
    # By role and correctness
    for role in ['helpful', 'sneaky']:
        for correct in [True, False]:
            mask = (roles == role) & (correctness == correct)
            if np.any(mask):
                key = f"{role}_{correct}"
                stats['by_role_and_correctness'][key] = {
                    'mean': float(np.mean(rewards[mask])),
                    'std': float(np.std(rewards[mask])),
                    'count': int(np.sum(mask))
                }
    
    return stats
