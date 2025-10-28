"""
Simple VLLM Data Parallel Sampling - Based on Official VLLM Example
https://vllm.website.cncfstack.com/examples/offline_inference/data_parallel.html
"""

import os
import logging
from tqdm import tqdm
import json
import tempfile
import time
import threading
import queue
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from multiprocessing import Process, set_start_method, Queue, Event
from time import sleep
from vllm.outputs import RequestOutput
import uuid

from vllm import LLM, SamplingParams

from ..parsing import extract_final_answer, is_correct
# Import prompt formatting function
from ..prompts import format_simple_prompt
from ..runtime import TaskQueue, PersistentWorkerPool
logger = logging.getLogger(__name__)

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass


def fast_sample_solutions_for_verifier(
    problems: List[Dict[str, Any]],
    prover_model_name: str,
    k_per_role: int,
    round_num: int,
    device: str = "cuda",
    task_batch_size: int = 8,
    use_vllm: bool = True,
    data_parallel_size: int = None,
    gpu_memory_utilization: float = 0.8,
    gpu_ids: Optional[List[int]] = None,
    avoid_conflicts: bool = False,
    enable_timing: bool = True,
    max_tries_per_task: int = 5,
    output_dir: str = ".",
) -> List[Dict[str, Any]]:
    """
    Simple data parallel sampling with task queue for role-specific success criteria.
    
    - Helpful Prover: Success = Correct solution, Failure = Wrong solution → Resample
    - Sneaky Prover: Success = Wrong solution, Failure = Correct solution → Resample
    
    Args:
        problems: List of problem dictionaries
        prover_model_name: Name of the prover model
        k_per_role: Number of solutions per role per problem
        round_num: Training round number
        device: Device to use (default: "cuda")
        task_batch_size: Batch size for tasks (default: 8)
        use_vllm: Whether to use VLLM (default: True)
        data_parallel_size: Number of data parallel workers (default: None, auto-detect)
        gpu_memory_utilization: GPU memory utilization (default: 0.8)
        gpu_ids: List of GPU IDs to use (default: None, use all available)
        avoid_conflicts: Whether to avoid GPU conflicts (default: False)
        enable_timing: Whether to enable detailed timing and performance metrics
        max_tries_per_task: Maximum number of attempts per task (default: 5)
        output_dir: Directory to save the detailed report (default: ".")
        
    Returns:
        List of sampled solutions
    """
    # Handle data_parallel_size parameter
    if data_parallel_size is None:
        if gpu_ids:
            data_parallel_size = len(gpu_ids)
        else:
            data_parallel_size = 2  # Default fallback
    
    overall_start_time = time.time() if enable_timing else None
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if enable_timing else None
    
    if enable_timing:
        logger.info(f"[{timestamp}] Starting task queue-based data parallel sampling")
    else:
        logger.info("Starting task queue-based data parallel sampling")
    logger.info(f"Model: {prover_model_name}")
    logger.info(f"Problems: {len(problems)}")
    logger.info(f"Target: {k_per_role} solutions per role per problem")
    logger.info(f"Max tries per task: {max_tries_per_task}")
    logger.info(f"Workers: {data_parallel_size}")
    
    # Set CUDA_VISIBLE_DEVICES if gpu_ids is provided
    if gpu_ids:
        gpu_str = ",".join(map(str, gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_str}")
    else:
        logger.info("Using all available GPUs")
    
    # Define success function outside of TaskQueue (role-aware)
    def is_success_fn(solution: Dict[str, Any]) -> bool:
        role = solution.get('role')
        is_corr = solution.get('is_correct')
        if role == 'helpful':
            return bool(is_corr)
        if role == 'sneaky':
            return not bool(is_corr)
        # Default: treat correctness as success if role missing
        return bool(is_corr)

    # Initialize task queue with callback
    task_queue = TaskQueue(is_success_fn=is_success_fn, max_tries_per_task=max_tries_per_task)
    
    # Add initial tasks
    setup_start = time.time() if enable_timing else None
    for problem_idx, problem in enumerate(problems):
        problem_text = problem['problem']
        ground_truth = problem['ground_truth']
        
        # Create tasks for both roles
        for role in ['helpful', 'sneaky']:
            for _ in range(k_per_role):
                task_queue.add_task({
                    'problem_idx': problem_idx,
                    'problem_text': problem_text,
                    'ground_truth': ground_truth,
                    'role': role,
                    'prompt': format_simple_prompt(problem_text, role),
                    'tries': 0
                })
    
    setup_time = time.time() - setup_start if enable_timing else None
    logger.info(f"Added {task_queue.task_counter} initial tasks in {setup_time:.2f}s" if enable_timing else f"Added {task_queue.task_counter} initial tasks")
    
    # Initialize persistent worker pool
    worker_pool = PersistentWorkerPool(
        num_workers=data_parallel_size,
        model_name=prover_model_name,
        gpu_ids=gpu_ids,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_timing=enable_timing,
        worker_batch_size=task_batch_size,
        show_progress=False
    )
    
    try:
        # Progress bar tracking successful tasks over all initial tasks
        total_initial_tasks = task_queue.task_counter
        pbar = tqdm(total=total_initial_tasks, desc="Successful tasks", leave=True)
        pbar.update(0)

        # Start persistent workers
        worker_pool.start_workers()
        
        # Main sampling loop with persistent workers
        batch_size = data_parallel_size * task_batch_size  # Adjust batch size based on workers
        
        while task_queue.should_continue():
            iteration_start = time.time() if enable_timing else None
            
            # Get next batch of tasks
            batch_tasks = task_queue.get_next_batch(batch_size)
            
            if not batch_tasks:
                logger.warning("No tasks available for processing")
                break
            
            logger.info(f"Processing {len(batch_tasks)} tasks")
            
            # Submit tasks to persistent worker pool
            worker_pool.submit_tasks(batch_tasks)
            
            # Wait for all tasks in this batch to complete
            batch_solutions = worker_pool.wait_for_completion(
                expected_results=len(batch_tasks),
                timeout=300.0  # 5 minute timeout per batch
            )
            
            # Process each solution
            successful_count = 0
            failed_count = 0
            
            for solution in batch_solutions:
                # Ensure required metadata
                assert 'problem_idx' in solution
                # Parse and score outside the worker
                generated_text = solution.get('solution', '')
                predicted_answer = extract_final_answer(generated_text)
                solution['predicted_answer'] = predicted_answer
                # Compute correctness using ground truth from the original task
                solution['is_correct'] = is_correct(predicted_answer, solution.get('ground_truth', ''))
                
                is_successful = task_queue.process_solution(solution)
                if is_successful:
                    successful_count += 1
                else:
                    failed_count += 1
            
            iteration_time = time.time() - iteration_start if enable_timing else None
            
            # Get current status
            status = task_queue.get_status()
            
            logger.info(f"Batch completed:")
            logger.info(f"  - Generated: {len(batch_solutions)} solutions")
            logger.info(f"  - Successful: {successful_count}")
            logger.info(f"  - Failed (will resample): {failed_count}")
            logger.info(f"  - Total completed: {status['completed_solutions']}")
            logger.info(f"  - Total attempts: {status['total_attempts']}")
            # Success rates by role are now tracked outside TaskQueue; omitted here
            if enable_timing:
                logger.info(f"  - Batch time: {iteration_time:.2f}s")

            # Update progress bar with new successful count
            current_completed = status['completed_solutions']
            if pbar.n < current_completed:
                pbar.update(current_completed - pbar.n)
            pbar.set_postfix_str(f"{current_completed}/{total_initial_tasks}")
            
    
    finally:
        # Always shutdown workers
        worker_pool.shutdown()
        try:
            pbar.close()
        except Exception:
            pass
    
    # Final status
    final_status = task_queue.get_status()
    total_time = time.time() - overall_start_time if enable_timing else None
    
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Completed solutions: {final_status['completed_solutions']}")
    logger.info(f"Total attempts: {final_status['total_attempts']}")
    if enable_timing:
        logger.info(f"Total time: {total_time:.2f}s")
    logger.info("=" * 60)
    
    logger.info("Task queue-based data parallel sampling completed successfully")
    
    # Generate detailed report
    report_path = task_queue.generate_report(output_dir)
    logger.info(f"Detailed report saved to: {report_path}")
    
    return task_queue.completed_solutions

# Example usage function
def example_usage():
    """Example of how to use the task queue-based data parallel sampling."""
    
    # Configure logging to see the output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Sample problems - smaller set for testing
    problems = []
    for i in range(5):  # Reduced for faster testing
        problems.append({
            'problem': f'What is {i+1}+{i+1}?',
            'ground_truth': str((i+1)*2)
        })
    
    # Run sampling with task queue
    solutions = fast_sample_solutions_for_verifier(
        problems=problems,
        prover_model_name="Qwen/Qwen2.5-0.5B",
        k_per_role=2,
        round_num=1,
        device="cuda",
        task_batch_size=4,  # Smaller batch size for testing
        use_vllm=True,
        data_parallel_size=2,  # Use 2 GPUs
        gpu_memory_utilization=0.1,
        gpu_ids=[5, 6],  # Use available GPUs
        avoid_conflicts=False,
        enable_timing=True,  # Enable detailed timing
        max_tries_per_task=3,  # Limit tries for testing
        output_dir=".",  # Save report in current directory
    )
    
    print(f"Generated {len(solutions)} solutions")
    
    # Analyze results by role
    helpful_solutions = [s for s in solutions if s['role'] == 'helpful']
    sneaky_solutions = [s for s in solutions if s['role'] == 'sneaky']
    
    helpful_correct = sum(1 for s in helpful_solutions if s['is_correct'])
    sneaky_wrong = sum(1 for s in sneaky_solutions if not s['is_correct'])
    
    print(f"Helpful solutions: {len(helpful_solutions)} (correct: {helpful_correct})")
    print(f"Sneaky solutions: {len(sneaky_solutions)} (wrong: {sneaky_wrong})")

if __name__ == "__main__":
    example_usage()
