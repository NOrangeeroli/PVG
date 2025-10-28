"""
Evaluation metrics for measuring model performance.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import torch
import logging

logger = logging.getLogger(__name__)

def calculate_accuracy(predictions: List[bool], ground_truth: List[bool]) -> float:
    """Calculate accuracy score."""
    return accuracy_score(ground_truth, predictions)

def calculate_precision(predictions: List[bool], ground_truth: List[bool]) -> float:
    """Calculate precision score."""
    return precision_score(ground_truth, predictions, zero_division=0)

def calculate_recall(predictions: List[bool], ground_truth: List[bool]) -> float:
    """Calculate recall score."""
    return recall_score(ground_truth, predictions, zero_division=0)

def calculate_f1(predictions: List[bool], ground_truth: List[bool]) -> float:
    """Calculate F1 score."""
    return f1_score(ground_truth, predictions, zero_division=0)

def calculate_auc(scores: List[float], ground_truth: List[bool]) -> float:
    """Calculate ROC-AUC score."""
    try:
        return roc_auc_score(ground_truth, scores)
    except ValueError:
        return 0.5  # Default for binary classification

def calculate_pr_auc(scores: List[float], ground_truth: List[bool]) -> float:
    """Calculate PR-AUC score."""
    try:
        precision, recall, _ = precision_recall_curve(ground_truth, scores)
        return auc(recall, precision)
    except ValueError:
        return 0.5

def calculate_ece(
    scores: List[float], 
    ground_truth: List[bool], 
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        scores: List of confidence scores
        ground_truth: List of true labels
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    scores = np.array(scores)
    ground_truth = np.array(ground_truth)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (scores > bin_lower) & (scores <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence in this bin
            accuracy_in_bin = ground_truth[in_bin].mean()
            avg_confidence_in_bin = scores[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)

def calculate_mce(
    scores: List[float], 
    ground_truth: List[bool], 
    n_bins: int = 10
) -> float:
    """
    Calculate Maximum Calibration Error (MCE).
    
    Args:
        scores: List of confidence scores
        ground_truth: List of true labels
        n_bins: Number of bins for calibration
        
    Returns:
        MCE value
    """
    scores = np.array(scores)
    ground_truth = np.array(ground_truth)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (scores > bin_lower) & (scores <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence in this bin
            accuracy_in_bin = ground_truth[in_bin].mean()
            avg_confidence_in_bin = scores[in_bin].mean()
            
            # Update MCE
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return float(mce)

def calculate_confusion_matrix_metrics(
    predictions: List[bool], 
    ground_truth: List[bool]
) -> Dict[str, int]:
    """Calculate confusion matrix metrics."""
    cm = confusion_matrix(ground_truth, predictions)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
    else:
        # Handle edge cases
        return {
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0
        }

def calculate_all_metrics(
    predictions: List[bool],
    ground_truth: List[bool],
    scores: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        scores: Optional list of confidence scores
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': calculate_accuracy(predictions, ground_truth),
        'precision': calculate_precision(predictions, ground_truth),
        'recall': calculate_recall(predictions, ground_truth),
        'f1': calculate_f1(predictions, ground_truth)
    }
    
    # Add confusion matrix metrics
    cm_metrics = calculate_confusion_matrix_metrics(predictions, ground_truth)
    metrics.update(cm_metrics)
    
    # Add calibration metrics if scores provided
    if scores is not None:
        metrics.update({
            'auc': calculate_auc(scores, ground_truth),
            'pr_auc': calculate_pr_auc(scores, ground_truth),
            'ece': calculate_ece(scores, ground_truth),
            'mce': calculate_mce(scores, ground_truth)
        })
    
    return metrics

def calculate_per_role_metrics(
    predictions: List[bool],
    ground_truth: List[bool],
    roles: List[str],
    scores: Optional[List[float]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics separately for each role.
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        roles: List of roles
        scores: Optional list of confidence scores
        
    Returns:
        Dictionary with metrics for each role
    """
    role_metrics = {}
    
    for role in set(roles):
        # Get indices for this role
        role_indices = [i for i, r in enumerate(roles) if r == role]
        
        if not role_indices:
            continue
        
        # Get predictions and ground truth for this role
        role_predictions = [predictions[i] for i in role_indices]
        role_ground_truth = [ground_truth[i] for i in role_indices]
        role_scores = [scores[i] for i in role_indices] if scores else None
        
        # Calculate metrics
        role_metrics[role] = calculate_all_metrics(
            role_predictions, 
            role_ground_truth, 
            role_scores
        )
    
    return role_metrics

def calculate_legibility_metrics(
    solutions: List[str],
    ground_truth: List[str],
    roles: List[str]
) -> Dict[str, float]:
    """
    Calculate legibility metrics for solutions.
    
    Args:
        solutions: List of generated solutions
        ground_truth: List of ground truth answers
        roles: List of roles
        
    Returns:
        Dictionary with legibility metrics
    """
    from ..parsing import extract_final_answer, extract_solution_steps, extract_reasoning_quality_indicators
    
    legibility_metrics = {
        'avg_solution_length': 0.0,
        'avg_steps_per_solution': 0.0,
        'avg_quality_score': 0.0,
        'helpful_legibility': 0.0,
        'sneaky_legibility': 0.0
    }
    
    if not solutions:
        return legibility_metrics
    
    # Calculate basic metrics
    solution_lengths = [len(s.split()) for s in solutions]
    legibility_metrics['avg_solution_length'] = float(np.mean(solution_lengths))
    
    # Calculate steps per solution
    steps_per_solution = []
    for solution in solutions:
        steps = extract_solution_steps(solution)
        steps_per_solution.append(len(steps))
    legibility_metrics['avg_steps_per_solution'] = float(np.mean(steps_per_solution))
    
    # Calculate quality indicators
    quality_scores = []
    for solution in solutions:
        indicators = extract_reasoning_quality_indicators(solution)
        quality_score = sum(indicators.values()) / len(indicators)
        quality_scores.append(quality_score)
    legibility_metrics['avg_quality_score'] = float(np.mean(quality_scores))
    
    # Calculate role-specific legibility
    for role in ['helpful', 'sneaky']:
        role_indices = [i for i, r in enumerate(roles) if r == role]
        if role_indices:
            role_solutions = [solutions[i] for i in role_indices]
            role_quality_scores = [quality_scores[i] for i in role_indices]
            legibility_metrics[f'{role}_legibility'] = float(np.mean(role_quality_scores))
    
    return legibility_metrics

def calculate_reward_metrics(
    rewards: List[float],
    roles: List[str],
    ground_truth: List[bool]
) -> Dict[str, float]:
    """
    Calculate reward distribution metrics.
    
    Args:
        rewards: List of reward values
        roles: List of roles
        ground_truth: List of correctness labels
        
    Returns:
        Dictionary with reward metrics
    """
    reward_metrics = {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards))
    }
    
    # Calculate reward by role
    for role in ['helpful', 'sneaky']:
        role_indices = [i for i, r in enumerate(roles) if r == role]
        if role_indices:
            role_rewards = [rewards[i] for i in role_indices]
            reward_metrics[f'{role}_mean_reward'] = float(np.mean(role_rewards))
            reward_metrics[f'{role}_std_reward'] = float(np.std(role_rewards))
    
    # Calculate reward by correctness
    for correct in [True, False]:
        correct_indices = [i for i, c in enumerate(ground_truth) if c == correct]
        if correct_indices:
            correct_rewards = [rewards[i] for i in correct_indices]
            reward_metrics[f'correct_{correct}_mean_reward'] = float(np.mean(correct_rewards))
            reward_metrics[f'correct_{correct}_std_reward'] = float(np.std(correct_rewards))
    
    return reward_metrics

def calculate_comprehensive_metrics(
    predictions: List[bool],
    ground_truth: List[bool],
    scores: Optional[List[float]] = None,
    solutions: Optional[List[str]] = None,
    roles: Optional[List[str]] = None,
    rewards: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        scores: Optional list of confidence scores
        solutions: Optional list of generated solutions
        roles: Optional list of roles
        rewards: Optional list of reward values
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'basic_metrics': calculate_all_metrics(predictions, ground_truth, scores)
    }
    
    # Add per-role metrics if roles provided
    if roles is not None:
        metrics['per_role_metrics'] = calculate_per_role_metrics(
            predictions, ground_truth, roles, scores
        )
    
    # Add legibility metrics if solutions provided
    if solutions is not None and roles is not None:
        metrics['legibility_metrics'] = calculate_legibility_metrics(
            solutions, ground_truth, roles
        )
    
    # Add reward metrics if rewards provided
    if rewards is not None and roles is not None:
        metrics['reward_metrics'] = calculate_reward_metrics(
            rewards, roles, ground_truth
        )
    
    return metrics
