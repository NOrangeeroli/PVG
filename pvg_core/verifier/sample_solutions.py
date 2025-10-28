"""
Solution sampling for verifier training.

Implements the paper's approach: for each problem, sample k correct solutions 
from helpful prover and k incorrect solutions from sneaky prover.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from tqdm import tqdm

from ..parsing import extract_final_answer, is_correct
from ..prompts import format_simple_prompt

# Try to import fast sampling
try:
    from .fast_sampling import fast_sample_solutions_for_verifier
    FAST_SAMPLING_AVAILABLE = True
except ImportError:
    FAST_SAMPLING_AVAILABLE = False

# Simple data parallel sampling is now part of fast_sampling.py
SIMPLE_DP_AVAILABLE = FAST_SAMPLING_AVAILABLE

logger = logging.getLogger(__name__)

def sample_from_prover(
    problems: List[str],
    prover_model,
    tokenizer,
    role: str,
    k_target: int,
    max_tries_per_task: int,
    ground_truths: List[str],
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """
    Sample solutions from a prover with a specific role until target count is reached.
    
    Args:
        problems: List of math problems
        prover_model: The prover model to sample from
        tokenizer: Tokenizer for the model
        role: "helpful" or "sneaky"
        k_target: Target number of solutions to collect
        max_tries_per_task: Maximum number of attempts per task before giving up
        ground_truths: List of ground truth answers
        device: Device to use
        
    Returns:
        List of sampled solutions with metadata
    """
    solutions = []
    problem_indices = list(range(len(problems)))
    
    # Sample until we get enough solutions
    attempts = 0
    while len(solutions) < k_target and attempts < max_tries_per_task:
        # Sample from remaining problems
        if not problem_indices:
            break
            
        # Select a random problem
        problem_idx = np.random.choice(problem_indices)
        problem = problems[problem_idx]
        ground_truth = ground_truths[problem_idx]
        
        # Format prompt
        prompt = format_simple_prompt(problem, role)
        
        # Tokenize and generate
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(device)
        
        with torch.no_grad():
            outputs = prover_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode solution
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        solution = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Check correctness
        predicted_answer = extract_final_answer(solution)
        is_solution_correct = is_correct(predicted_answer, ground_truth)
        
        # Determine if this solution meets our criteria
        if role == "helpful" and is_solution_correct:
            # We want correct solutions from helpful prover
            solutions.append({
                'problem': problem,
                'solution': solution,
                'ground_truth': ground_truth,
                'is_correct': True,
                'role': role,
                'predicted_answer': predicted_answer
            })
        elif role == "sneaky" and not is_solution_correct:
            # We want incorrect solutions from sneaky prover
            solutions.append({
                'problem': problem,
                'solution': solution,
                'ground_truth': ground_truth,
                'is_correct': False,
                'role': role,
                'predicted_answer': predicted_answer
            })
        
        attempts += 1
    
    logger.info("Sampled %d %s solutions in %d attempts", len(solutions), role, attempts)
    return solutions

def sample_from_prover_for_single_problem(
    problem_text: str,
    ground_truth: str,
    prover_model,
    tokenizer,
    role: str,
    k_target: int,
    max_tries_per_task: int,
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """
    Sample solutions from a prover for a single problem until target count is reached.
    
    Args:
        problem_text: The math problem text
        ground_truth: The correct answer
        prover_model: The prover model to sample from
        tokenizer: Tokenizer for the model
        role: "helpful" or "sneaky"
        k_target: Target number of solutions to collect
        max_tries_per_task: Maximum number of attempts per task before giving up
        device: Device to use
        
    Returns:
        List of sampled solutions with metadata
    """
    solutions = []
    attempts = 0
    
    # Create progress bar for sampling attempts
    attempt_pbar = tqdm(
        total=k_target,
        desc=f"Sampling {role} solutions",
        unit="solution",
        leave=False  # Don't leave progress bar after completion
    )
    
    try:
        while len(solutions) < k_target and attempts < max_tries_per_task:
            # Format prompt for this specific problem
            prompt = format_simple_prompt(problem_text, role)
            
            # Tokenize and generate
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(device)
            
            with torch.no_grad():
                outputs = prover_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode solution
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            solution = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Check correctness
            predicted_answer = extract_final_answer(solution)
            is_solution_correct = is_correct(predicted_answer, ground_truth)
            
            # Determine if this solution meets our criteria
            solution_added = False
            if role == "helpful" and is_solution_correct:
                # We want correct solutions from helpful prover
                solutions.append({
                    'problem': problem_text,
                    'solution': solution,
                    'ground_truth': ground_truth,
                    'is_correct': True,
                    'role': role,
                    'predicted_answer': predicted_answer
                })
                solution_added = True
            elif role == "sneaky" and not is_solution_correct:
                # We want incorrect solutions from sneaky prover
                solutions.append({
                    'problem': problem_text,
                    'solution': solution,
                    'ground_truth': ground_truth,
                    'is_correct': False,
                    'role': role,
                    'predicted_answer': predicted_answer
                })
                solution_added = True
            
            attempts += 1
            
            # Update progress bar
            if solution_added:
                attempt_pbar.update(1)
            
            attempt_pbar.set_postfix({
                'found': len(solutions),
                'target': k_target,
                'attempts': attempts
            })
    finally:
        # Always close progress bar, even on exception
        attempt_pbar.close()
    
    logger.info("Problem: sampled %d %s solutions in %d attempts", len(solutions), role, attempts)
    return solutions

def sample_solutions_for_verifier(
    problems: List[Dict[str, Any]],
    prover_model,
    tokenizer,
    k_per_role: int,
    max_tries_per_task: int,
    round_num: int,
    device: str = "cuda",
    use_fast_sampling: bool = True,
    task_batch_size: int = 8,
    data_parallel_size: int = None,
    gpu_memory_utilization: float = 0.8,
    prover_model_name: str = None,
    gpu_ids: Optional[List[int]] = None,
    avoid_conflicts: bool = False,
    output_dir: str = "."
) -> List[Dict[str, Any]]:
    """
    Sample solutions from helpful/sneaky provers for verifier training.
    
    For each problem:
    1. Sample k correct solutions from helpful prover
    2. Sample k incorrect solutions from sneaky prover
    3. Tag with is_correct and role
    4. Cache results
    
    Args:
        problems: List of problem dictionaries with 'problem' and 'ground_truth' keys
        prover_model: The prover model to sample from
        tokenizer: Tokenizer for the model
        k_per_role: Number of solutions per role per problem
        max_tries_per_task: Maximum number of attempts per task before giving up
        round_num: Training round number
        device: Device to use
        use_fast_sampling: Whether to use VLLM fast sampling
        task_batch_size: Number of tasks to process in each batch
        data_parallel_size: Number of GPUs for data parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
        prover_model_name: Name of the prover model
        gpu_ids: Specific GPUs to use
        avoid_conflicts: Whether to avoid GPU conflicts
        output_dir: Directory to save detailed reports
        
    Returns:
        List of sampled solutions with is_correct labels
    """
    logger.info("Sampling solutions for verifier training (round %d)", round_num)
    logger.info("Target: %d solutions per role per problem", k_per_role)
    
    # Try simple data parallel sampling first if available and enabled
    if use_fast_sampling and SIMPLE_DP_AVAILABLE and data_parallel_size and data_parallel_size > 1:
        logger.info("Using simple data parallel sampling with VLLM")
        try:
            # Use provided model name or try to extract from prover_model
            if prover_model_name:
                model_name = prover_model_name
                logger.info("Using provided model name: %s", model_name)
            elif hasattr(prover_model, 'config') and hasattr(prover_model.config, 'name_or_path'):
                model_name = prover_model.config.name_or_path
            elif hasattr(prover_model, 'config') and hasattr(prover_model.config, '_name_or_path'):
                model_name = prover_model.config._name_or_path
            else:
                # Fallback: try to get from the model's class or use a default
                model_name = getattr(prover_model, '__class__', {}).__name__ if hasattr(prover_model, '__class__') else 'unknown'
                logger.warning("Could not determine model name from config, using: %s", model_name)
            
            return fast_sample_solutions_for_verifier(
                problems=problems,
                prover_model_name=model_name,
                k_per_role=k_per_role,
                round_num=round_num,
                data_parallel_size=data_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                gpu_ids=gpu_ids,
                max_tries_per_task=max_tries_per_task,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.error("Fast sampling failed: %s", e)
            logger.error("Falling back to standard sampling is disabled for debugging")
            raise Exception(f"Fast sampling failed and fallback is disabled: {e}")
    
    # Try complex fast sampling as fallback
    elif use_fast_sampling and FAST_SAMPLING_AVAILABLE:
        logger.info("Using complex fast sampling with VLLM")
        try:
            # Use provided model name or try to extract from prover_model
            if prover_model_name:
                model_name = prover_model_name
                logger.info("Using provided model name: %s", model_name)
            elif hasattr(prover_model, 'config') and hasattr(prover_model.config, 'name_or_path'):
                model_name = prover_model.config.name_or_path
            elif hasattr(prover_model, 'config') and hasattr(prover_model.config, '_name_or_path'):
                model_name = prover_model.config._name_or_path
            else:
                # Fallback: try to get from the model's class or use a default
                model_name = getattr(prover_model, '__class__', {}).__name__ if hasattr(prover_model, '__class__') else 'unknown'
                logger.warning("Could not determine model name from config, using: %s", model_name)
            return fast_sample_solutions_for_verifier(
                problems=problems,
                prover_model_name=model_name,
                k_per_role=k_per_role,
                max_tries_per_task=max_tries_per_task,
                round_num=round_num,
                device=device,
                task_batch_size=task_batch_size,
                use_vllm=True,
                data_parallel_size=data_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                gpu_ids=gpu_ids,
                avoid_conflicts=avoid_conflicts,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.error("Complex fast sampling failed: %s", e)
            logger.error("Falling back to standard sampling is disabled for debugging")
            raise Exception(f"Complex fast sampling failed and fallback is disabled: {e}")
    
    logger.info("Using standard sampling")
    
    all_solutions = []
    
    # Process each problem individually with progress bar
    problem_pbar = tqdm(
        enumerate(problems), 
        total=len(problems),
        desc="Sampling solutions",
        unit="problem"
    )
    
    for problem_idx, problem in problem_pbar:
        problem_text = problem['problem']
        ground_truth = problem['ground_truth']
        
        # Update progress bar description
        problem_pbar.set_description(f"Problem {problem_idx + 1}/{len(problems)}")
        
        # Sample k correct solutions from helpful prover for this problem
        helpful_solutions = sample_from_prover_for_single_problem(
            problem_text=problem_text,
            ground_truth=ground_truth,
            prover_model=prover_model,
            tokenizer=tokenizer,
            role="helpful",
            k_target=k_per_role,
            max_tries_per_task=max_tries_per_task,
            device=device
        )
        all_solutions.extend(helpful_solutions)
        
        # Sample k incorrect solutions from sneaky prover for this problem
        sneaky_solutions = sample_from_prover_for_single_problem(
            problem_text=problem_text,
            ground_truth=ground_truth,
            prover_model=prover_model,
            tokenizer=tokenizer,
            role="sneaky",
            k_target=k_per_role,
            max_tries_per_task=max_tries_per_task,
            device=device
        )
        all_solutions.extend(sneaky_solutions)
        
        # Update progress bar with current stats
        correct_count = sum(1 for s in all_solutions if s['is_correct'])
        incorrect_count = len(all_solutions) - correct_count
        problem_pbar.set_postfix({
            'total': len(all_solutions),
            'correct': correct_count,
            'incorrect': incorrect_count
        })
    
    # Log statistics
    correct_count = sum(1 for s in all_solutions if s['is_correct'])
    incorrect_count = len(all_solutions) - correct_count
    logger.info("Total solutions: %d (correct: %d, incorrect: %d)", len(all_solutions), correct_count, incorrect_count)
    
    return all_solutions

def cache_solutions(
    solutions: List[Dict[str, Any]],
    cache_file: str
) -> None:
    """Save sampled solutions to disk."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(solutions, f, indent=2)
    
    logger.info("Cached %d solutions to %s", len(solutions), cache_file)

def load_cached_solutions(cache_file: str) -> Optional[List[Dict[str, Any]]]:
    """Load previously sampled solutions from disk."""
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            solutions = json.load(f)
        logger.info("Loaded %d cached solutions from %s", len(solutions), cache_file)
        return solutions
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Failed to load cached solutions from %s: %s", cache_file, e)
        return None

def validate_solutions(solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate and analyze sampled solutions."""
    if not solutions:
        return {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'helpful': 0,
            'sneaky': 0,
            'balance_ratio': 0.0
        }
    
    total = len(solutions)
    correct = sum(1 for s in solutions if s['is_correct'])
    incorrect = total - correct
    helpful = sum(1 for s in solutions if s['role'] == 'helpful')
    sneaky = total - helpful
    
    balance_ratio = min(correct, incorrect) / max(correct, incorrect) if max(correct, incorrect) > 0 else 0.0
    
    stats = {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'helpful': helpful,
        'sneaky': sneaky,
        'balance_ratio': balance_ratio
    }
    
    logger.info("Solution validation: %s", stats)
    return stats
