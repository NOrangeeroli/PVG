"""
Simple VLLM Data Parallel Sampling - Based on Official VLLM Example
https://vllm.website.cncfstack.com/examples/offline_inference/data_parallel.html
"""

import os
import logging
import json
import tempfile
import time
import threading
import queue
from datetime import datetime
from typing import List, Dict, Any, Optional
from multiprocessing import Process, set_start_method, Queue, Event
from time import sleep
from vllm.outputs import RequestOutput
import uuid

from vllm import LLM, SamplingParams

from ..parsing import extract_final_answer, is_correct
# Import prompt formatting function
from ..prompts import format_simple_prompt
logger = logging.getLogger(__name__)

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

class TaskQueue:
    """
    Task queue for managing sampling tasks with role-specific success criteria.
    
    - Helpful Prover: Success = Correct solution, Failure = Wrong solution → Resample
    - Sneaky Prover: Success = Wrong solution, Failure = Correct solution → Resample
    """
    
    def __init__(self, max_tries_per_task: int = 5):
        self.max_tries_per_task = max_tries_per_task
        self.pending_tasks = []  # Tasks waiting to be processed
        self.completed_solutions = []  # Successfully completed solutions
        self.failed_tasks = []  # Tasks that failed and need resampling
        self.task_counter = 0
        self.task_history = []  # Track all task attempts for reporting
        
        # Track success rates by role
        self.helpful_success_count = 0
        self.helpful_total_count = 0
        self.sneaky_success_count = 0
        self.sneaky_total_count = 0
    
    def add_task(self, problem_idx: int, problem_text: str, ground_truth: str, role: str):
        """Add a new task to the queue"""
        task_id = str(uuid.uuid4())
        task = {
            'task_id': task_id,
            'problem_idx': problem_idx,
            'problem_text': problem_text,
            'ground_truth': ground_truth,
            'role': role,
            'tries': 0
        }
        self.pending_tasks.append(task)
        self.task_counter += 1
    
    def is_solution_successful(self, solution: Dict[str, Any]) -> bool:
        """Determine if a solution is successful based on role"""
        role = solution['role']
        is_correct = solution['is_correct']
        
        if role == 'helpful':
            return is_correct  # Success = correct solution
        elif role == 'sneaky':
            return not is_correct  # Success = wrong solution
        else:
            raise ValueError(f"Unknown role: {role}")
    
    def process_solution(self, solution: Dict[str, Any]) -> bool:
        """Process a solution and determine if it's successful"""
        is_successful = self.is_solution_successful(solution)
        
        # Find the original task to track tries
        task_id = solution.get('task_id')
        original_task = None
        for task in self.pending_tasks + self.failed_tasks:
            if task.get('task_id') == task_id:
                original_task = task
                break
        
        if original_task:
            original_task['tries'] += 1
            current_tries = original_task['tries']
        else:
            current_tries = 1
        
        # Record this attempt in history
        attempt_record = {
            'task_id': task_id,
            'problem_idx': solution.get('problem_idx', 0),
            'problem_text': solution['problem'],
            'ground_truth': solution['ground_truth'],
            'role': solution['role'],
            'tries': current_tries,
            'is_successful': is_successful,
            'is_correct': solution.get('is_correct', False),
            'predicted_answer': solution.get('predicted_answer', ''),
            'solution': solution.get('solution', '')
        }
        self.task_history.append(attempt_record)
        
        # Update role-specific counters
        if solution['role'] == 'helpful':
            self.helpful_total_count += 1
            if is_successful:
                self.helpful_success_count += 1
        elif solution['role'] == 'sneaky':
            self.sneaky_total_count += 1
            if is_successful:
                self.sneaky_success_count += 1
        
        if is_successful:
            self.completed_solutions.append(solution)
            logger.info(f"Successful {solution['role']} solution added (total: {len(self.completed_solutions)})")
            return True
        else:
            # Check if task has reached max tries
            if current_tries >= self.max_tries_per_task:
                logger.info(f"Task {task_id} reached max tries ({self.max_tries_per_task}), removing from queue")
                # Remove from pending/failed tasks
                if original_task in self.pending_tasks:
                    self.pending_tasks.remove(original_task)
                if original_task in self.failed_tasks:
                    self.failed_tasks.remove(original_task)
                return False
            
            # Add failed task back to queue for resampling
            failed_task = {
                'task_id': task_id,
                'problem_idx': solution.get('problem_idx', 0),
                'problem_text': solution['problem'],
                'ground_truth': solution['ground_truth'],
                'role': solution['role'],
                'tries': current_tries
            }
            self.failed_tasks.append(failed_task)
            logger.info(f"Failed {solution['role']} solution added to resample queue (tries: {current_tries}/{self.max_tries_per_task})")
            return False
    
    def get_next_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get next batch of tasks for processing"""
        # First, add failed tasks from previous iteration to pending
        self.pending_tasks.extend(self.failed_tasks)
        self.failed_tasks.clear()
        
        # Get batch from pending tasks
        batch = self.pending_tasks[:batch_size]
        self.pending_tasks = self.pending_tasks[batch_size:]
        
        return batch
    
    def should_continue(self) -> bool:
        """Check if we should continue sampling"""
        if len(self.pending_tasks) == 0 and len(self.failed_tasks) == 0:
            logger.warning("No more tasks to process, stopping")
            return False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the task queue"""
        helpful_success_rate = (self.helpful_success_count / self.helpful_total_count * 100) if self.helpful_total_count > 0 else 0
        sneaky_success_rate = (self.sneaky_success_count / self.sneaky_total_count * 100) if self.sneaky_total_count > 0 else 0
        
        return {
            'completed_solutions': len(self.completed_solutions),
            'pending_tasks': len(self.pending_tasks),
            'failed_tasks': len(self.failed_tasks),
            'max_tries_per_task': self.max_tries_per_task,
            'helpful_success_rate': helpful_success_rate,
            'sneaky_success_rate': sneaky_success_rate,
            'total_attempts': len(self.task_history)
        }
    
    def generate_report(self, output_dir: str = ".") -> str:
        """Generate detailed report of all task attempts and save to file"""
        import os
        from datetime import datetime
        
        # Group attempts by task_id to get final status
        task_summaries = {}
        for attempt in self.task_history:
            task_id = attempt['task_id']
            if task_id not in task_summaries:
                task_summaries[task_id] = {
                    'problem_idx': attempt['problem_idx'],
                    'problem_text': attempt['problem_text'],
                    'ground_truth': attempt['ground_truth'],
                    'role': attempt['role'],
                    'total_tries': 0,
                    'max_tries_reached': False,
                    'final_success': False,
                    'attempts': []
                }
            
            task_summaries[task_id]['total_tries'] = max(task_summaries[task_id]['total_tries'], attempt['tries'])
            task_summaries[task_id]['attempts'].append(attempt)
            
            # Check if this was the final attempt
            if attempt['tries'] >= self.max_tries_per_task:
                task_summaries[task_id]['max_tries_reached'] = True
            
            # Check if this was successful
            if attempt['is_successful']:
                task_summaries[task_id]['final_success'] = True
        
        # Generate report content
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_content = []
        report_content.append("=" * 80)
        report_content.append("TASK QUEUE SAMPLING REPORT")
        report_content.append("=" * 80)
        report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"Max tries per task: {self.max_tries_per_task}")
        report_content.append(f"Total unique tasks: {len(task_summaries)}")
        report_content.append(f"Total attempts: {len(self.task_history)}")
        report_content.append(f"Successful solutions: {len(self.completed_solutions)}")
        report_content.append("")
        
        # Summary statistics
        successful_tasks = sum(1 for task in task_summaries.values() if task['final_success'])
        max_tries_reached = sum(1 for task in task_summaries.values() if task['max_tries_reached'])
        
        report_content.append("SUMMARY STATISTICS:")
        report_content.append(f"  - Tasks that succeeded: {successful_tasks}/{len(task_summaries)} ({successful_tasks/len(task_summaries)*100:.1f}%)")
        report_content.append(f"  - Tasks that reached max tries: {max_tries_reached}/{len(task_summaries)} ({max_tries_reached/len(task_summaries)*100:.1f}%)")
        report_content.append("")
        
        # Role-specific statistics
        helpful_tasks = [task for task in task_summaries.values() if task['role'] == 'helpful']
        sneaky_tasks = [task for task in task_summaries.values() if task['role'] == 'sneaky']
        
        helpful_successful = sum(1 for task in helpful_tasks if task['final_success'])
        sneaky_successful = sum(1 for task in sneaky_tasks if task['final_success'])
        
        report_content.append("ROLE-SPECIFIC STATISTICS:")
        report_content.append(f"  - Helpful tasks: {len(helpful_tasks)} total, {helpful_successful} successful ({helpful_successful/len(helpful_tasks)*100:.1f}%)")
        report_content.append(f"  - Sneaky tasks: {len(sneaky_tasks)} total, {sneaky_successful} successful ({sneaky_successful/len(sneaky_tasks)*100:.1f}%)")
        report_content.append("")
        
        # Detailed task report
        report_content.append("DETAILED TASK REPORT:")
        report_content.append("=" * 80)
        
        for task_id, task_summary in sorted(task_summaries.items(), key=lambda x: (x[1]['role'], x[1]['problem_idx'])):
            report_content.append(f"Task ID: {task_id}")
            report_content.append(f"  Problem {task_summary['problem_idx']} ({task_summary['role']})")
            report_content.append(f"  Ground Truth: {task_summary['ground_truth']}")
            report_content.append(f"  Total Tries: {task_summary['total_tries']}/{self.max_tries_per_task}")
            report_content.append(f"  Max Tries Reached: {'Yes' if task_summary['max_tries_reached'] else 'No'}")
            report_content.append(f"  Final Success: {'Yes' if task_summary['final_success'] else 'No'}")
            
            # List all attempts
            report_content.append("  Attempts:")
            for attempt in sorted(task_summary['attempts'], key=lambda x: x['tries']):
                status = "✓ SUCCESS" if attempt['is_successful'] else "✗ FAILED"
                correct_status = "correct" if attempt['is_correct'] else "incorrect"
                report_content.append(f"    Try {attempt['tries']}: {status} ({correct_status}) - {attempt['predicted_answer']}")
            
            report_content.append("")
        
        # Save report to file
        report_filename = f"task_queue_report_{timestamp}.txt"
        report_path = os.path.join(output_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        logger.info(f"Task queue report saved to: {report_path}")
        return report_path
    


class PersistentWorkerPool:
    """
    A pool of persistent VLLM workers that stay alive across multiple batches.
    Workers continuously process tasks from a shared queue until shutdown.
    """
    
    def __init__(self, 
                 num_workers: int,
                 prover_model_name: str,
                 gpu_ids: List[int],
                 gpu_memory_utilization: float = 0.05,
                 max_num_seqs: int = 64,
                 max_model_len: int = 4096,
                 enable_timing: bool = True):
        self.num_workers = num_workers
        self.prover_model_name = prover_model_name
        self.gpu_ids = gpu_ids
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.enable_timing = enable_timing
        
        # Communication queues
        self.task_queue = Queue()  # Tasks to be processed
        self.result_queue = Queue()  # Completed results
        self.shutdown_event = Event()  # Signal to stop workers
        
        # Worker processes
        self.workers = []
        self.worker_threads = []
        
        # Statistics
        self.total_tasks_processed = 0
        self.total_results_collected = 0
        
        logger.info(f"Initializing PersistentWorkerPool with {num_workers} workers")
    
    def start_workers(self):
        """Start all persistent worker processes"""
        logger.info("Starting persistent VLLM workers...")
        
        for worker_id in range(self.num_workers):
            # Assign GPU to this worker
            gpu_id = self.gpu_ids[worker_id] if worker_id < len(self.gpu_ids) else worker_id
            
            # Create worker process
            worker_process = Process(
                target=self._persistent_worker,
                args=(
                    worker_id,
                    gpu_id,
                    self.prover_model_name,
                    self.gpu_memory_utilization,
                    self.max_num_seqs,
                    self.max_model_len,
                    self.enable_timing,
                    self.task_queue,
                    self.result_queue,
                    self.shutdown_event,
                ),
                name=f"PersistentWorker-{worker_id}"
            )
            
            worker_process.start()
            self.workers.append(worker_process)
            logger.info(f"Started persistent worker {worker_id} on GPU {gpu_id}")
        
        # Give workers time to initialize
        time.sleep(5)
        logger.info("All persistent workers started and initialized")
    
    def submit_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        """Submit a batch of tasks to the worker pool"""
        logger.info(f"Submitting {len(tasks)} tasks to worker pool")
        
        for task in tasks:
            self.task_queue.put(task)
            self.total_tasks_processed += 1
    
    def collect_results(self, timeout: float = 30.0) -> List[Dict[str, Any]]:
        """Collect available results from workers"""
        results = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Non-blocking get with short timeout
                result = self.result_queue.get(timeout=0.1)
                results.append(result)
                self.total_results_collected += 1
            except queue.Empty:
                # No more results available
                break
        
        if results:
            logger.info(f"Collected {len(results)} results from worker pool")
        
        return results
    
    def wait_for_completion(self, expected_results: int, timeout: float = 300.0) -> List[Dict[str, Any]]:
        """Wait for all expected results to be completed"""
        logger.info(f"Waiting for {expected_results} results to complete...")
        
        all_results = []
        start_time = time.time()
        
        while len(all_results) < expected_results and time.time() - start_time < timeout:
            batch_results = self.collect_results(timeout=1.0)
            all_results.extend(batch_results)
            
            if len(all_results) < expected_results:
                logger.info(f"Progress: {len(all_results)}/{expected_results} results completed")
                time.sleep(0.5)  # Brief pause before checking again
        
        if len(all_results) >= expected_results:
            logger.info(f"Successfully collected all {len(all_results)} expected results")
        else:
            logger.warning(f"Timeout reached. Collected {len(all_results)}/{expected_results} results")
        
        return all_results
    
    def shutdown(self):
        """Shutdown all workers gracefully"""
        logger.info("Shutting down persistent worker pool...")
        
        # Signal workers to stop
        self.shutdown_event.set()
        
        # Wait for workers to finish current tasks
        for i, worker in enumerate(self.workers):
            logger.info(f"Waiting for worker {i} to finish...")
            worker.join(timeout=30)
            
            if worker.is_alive():
                logger.warning(f"Worker {i} didn't stop gracefully, terminating...")
                worker.terminate()
                worker.join()
        
        logger.info("All workers shut down")
    
    @staticmethod
    def _persistent_worker(worker_id: int,
                          gpu_id: int,
                          model_name: str,
                          gpu_memory_utilization: float,
                          max_num_seqs: int,
                          max_model_len: int,
                          enable_timing: bool,
                          task_queue: Queue,
                          result_queue: Queue,
                          shutdown_event: Event):
        """Persistent worker that stays alive and processes tasks continuously"""
        
        # Set GPU for this worker
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        logger.info(f"Persistent worker {worker_id} starting on GPU {gpu_id}")
        
        # Initialize VLLM once
        init_start = time.time() if enable_timing else None
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            enforce_eager=True,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        init_time = time.time() - init_start if enable_timing else None
        
        if enable_timing:
            logger.info(f"Persistent worker {worker_id} VLLM init took {init_time:.2f}s")
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop=["\n\n\n"]
        )
        
        logger.info(f"Persistent worker {worker_id} ready and waiting for tasks")
        
        # Main processing loop
        tasks_processed = 0
        while not shutdown_event.is_set():
            try:
                # Get task with timeout
                task = task_queue.get(timeout=1.0)
                
                # Process the task
                logger.info(f"Worker {worker_id} processing task: {task.get('role', 'unknown')} for problem {task.get('problem_idx', 'unknown')}")
                
                # Format prompt
                prompt = format_simple_prompt(task['problem_text'], task['role'])
                
                # Generate solution
                gen_start = time.time() if enable_timing else None
                outputs = llm.generate([prompt], sampling_params)
                gen_time = time.time() - gen_start if enable_timing else None
                
                # Process output
                if outputs and len(outputs) > 0:
                    generated_text = outputs[0].outputs[0].text
                    predicted_answer = extract_final_answer(generated_text)
                    is_solution_correct = is_correct(predicted_answer, task['ground_truth'])
                    
                    # Create result
                    result = {
                        'problem': task['problem_text'],
                        'solution': generated_text,
                        'ground_truth': task['ground_truth'],
                        'is_correct': is_solution_correct,
                        'role': task['role'],
                        'predicted_answer': predicted_answer,
                        'worker_id': worker_id,
                        'problem_idx': task['problem_idx'],
                        'task_id': task['task_id'],
                        'tries': task.get('tries', 0)
                    }
                    
                    if enable_timing:
                        result.update({
                            'generation_time': gen_time,
                            'tokens_generated': len(generated_text.split())
                        })
                    
                    # Send result back
                    result_queue.put(result)
                    tasks_processed += 1
                    
                    logger.info(f"Worker {worker_id} completed task (total: {tasks_processed})")
                else:
                    logger.error(f"Worker {worker_id} got empty output for task")
                
            except queue.Empty:
                # No tasks available, continue waiting
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error processing task: {e}")
                continue
        
        logger.info(f"Persistent worker {worker_id} shutting down after processing {tasks_processed} tasks")


def sample_worker(
    model_name: str,
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    prompts: List[str],
    prompt_metadata: List[Dict[str, Any]],
    sampling_params: SamplingParams,
    result_file: str,
    gpu_memory_utilization: float = 0.1,
    max_num_seqs: int = 64,
    max_model_len: int = 4096,
    enforce_eager: bool = True,
    trust_remote_code: bool = True,
    enable_timing: bool = True,
):
    """
    Worker process for data parallel sampling.
    Each worker processes a different subset of prompts.
    """
    worker_start_time = time.time() if enable_timing else None
    
    # Set VLLM data parallel environment variables
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    
    logger.info(f"Worker {global_dp_rank} starting with {len(prompts)} prompts")
    
    try:
        # Time VLLM initialization
        init_start = time.time() if enable_timing else None
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  # Single GPU per worker
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        init_time = time.time() - init_start if enable_timing else None
        if enable_timing:
            logger.info(f"Worker {global_dp_rank} VLLM init took {init_time:.2f}s")
        
        # Time generation
        gen_start = time.time() if enable_timing else None
        outputs = llm.generate(prompts, sampling_params)
        gen_time = time.time() - gen_start if enable_timing else None
        
        if enable_timing:
            # Calculate throughput metrics
            total_tokens_generated = sum(len(output.outputs[0].text.split()) for output in outputs)
            tokens_per_second = total_tokens_generated / gen_time if gen_time > 0 else 0
            prompts_per_second = len(prompts) / gen_time if gen_time > 0 else 0
            
            logger.info(f"Worker {global_dp_rank} generation took {gen_time:.2f}s")
            logger.info(f"Worker {global_dp_rank} throughput: {tokens_per_second:.1f} tokens/s, {prompts_per_second:.1f} prompts/s")
        
        # Process outputs and create solutions
        process_start = time.time() if enable_timing else None
        solutions = []
        for i, output in enumerate(outputs):
            if i < len(prompt_metadata):
                metadata = prompt_metadata[i]
                generated_text = output.outputs[0].text
                
                predicted_answer = extract_final_answer(generated_text)
                is_solution_correct = is_correct(predicted_answer, metadata['ground_truth'])
                
                solution_data = {
                    'problem': metadata['problem_text'],
                    'solution': generated_text,
                    'ground_truth': metadata['ground_truth'],
                    'is_correct': is_solution_correct,
                    'role': metadata['role'],
                    'predicted_answer': predicted_answer,
                    'worker_rank': global_dp_rank
                }
                
                if enable_timing:
                    solution_data.update({
                        'generation_time': gen_time,
                        'tokens_generated': len(generated_text.split())
                    })
                
                solutions.append(solution_data)
        
        process_time = time.time() - process_start if enable_timing else None
        total_time = time.time() - worker_start_time if enable_timing else None
        
        logger.info(f"Worker {global_dp_rank} completed {len(solutions)} solutions")
        if enable_timing:
            logger.info(f"Worker {global_dp_rank} timing: init={init_time:.2f}s, gen={gen_time:.2f}s, process={process_time:.2f}s, total={total_time:.2f}s")
        
        # Save results to file
        save_start = time.time() if enable_timing else None
        with open(result_file, 'w') as f:
            json.dump(solutions, f, indent=2)
        save_time = time.time() - save_start if enable_timing else None
        
        if enable_timing:
            logger.info(f"Worker {global_dp_rank} saved results to {result_file} in {save_time:.2f}s")
        else:
            logger.info(f"Worker {global_dp_rank} saved results to {result_file}")
        return solutions
        
    except Exception as e:
        logger.error(f"Worker {global_dp_rank} failed: {e}")
        raise

def distribute_prompts(prompts: List[str], metadata: List[Dict[str, Any]], dp_size: int, enable_timing: bool = True) -> tuple[List[List[str]], List[List[Dict[str, Any]]]]:
    """
    Distribute prompts and metadata evenly across data parallel workers.
    """
    dist_start = time.time() if enable_timing else None
    
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size
    
    def start(rank):
        return rank * floor + min(rank, remainder)
    
    distributed_prompts = []
    distributed_metadata = []
    for rank in range(dp_size):
        rank_prompts = prompts[start(rank):start(rank + 1)]
        rank_metadata = metadata[start(rank):start(rank + 1)]
        if len(rank_prompts) == 0:
            # Ensure each worker has at least one prompt
            rank_prompts = ["Placeholder"]
            rank_metadata = [{'problem_text': 'Placeholder', 'ground_truth': '0', 'role': 'helpful', 'problem_idx': 0}]
        distributed_prompts.append(rank_prompts)
        distributed_metadata.append(rank_metadata)
    
    if enable_timing:
        dist_time = time.time() - dist_start
        logger.info(f"Prompt distribution took {dist_time:.3f}s")
    
    return distributed_prompts, distributed_metadata

def process_batch_with_workers(
    batch_tasks: List[Dict[str, Any]],
    prover_model_name: str,
    data_parallel_size: int,
    gpu_memory_utilization: float,
    gpu_ids: Optional[List[int]],
    enable_timing: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process a batch of tasks using the existing multi-process logic.
    """
    if not batch_tasks:
        return []
    
    logger.info(f"Processing batch of {len(batch_tasks)} tasks")
    
    # Convert tasks to prompts
    prompts = []
    metadata = []
    for task in batch_tasks:
        prompt = format_simple_prompt(task['problem_text'], task['role'])
        prompts.append(prompt)
        metadata.append({
            'problem_idx': task['problem_idx'],
            'problem_text': task['problem_text'],
            'ground_truth': task['ground_truth'],
            'role': task['role'],
            'task_id': task['task_id']
        })
    
    # Distribute prompts across workers
    distributed_prompts, distributed_metadata = distribute_prompts(prompts, metadata, data_parallel_size, enable_timing)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        stop=["\n\n\n"]  # Stop at triple newlines
    )
    
    # Set up data parallel communication
    dp_master_ip = "127.0.0.1"  # Single node
    # Use dynamic port to avoid conflicts
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    dp_master_port = sock.getsockname()[1]
    sock.close()
    
    # Create temporary result files for each worker
    result_files = []
    for i in range(data_parallel_size):
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'_worker_{i}.json')
        temp_file.close()
        result_files.append(temp_file.name)
    
    # Start worker processes
    processes = []
    for local_dp_rank in range(data_parallel_size):
        global_dp_rank = local_dp_rank
        worker_prompts = distributed_prompts[local_dp_rank]
        worker_metadata = distributed_metadata[local_dp_rank]
        result_file = result_files[local_dp_rank]
        
        proc = Process(
            target=sample_worker,
            args=(
                prover_model_name,
                data_parallel_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                worker_prompts,
                worker_metadata,
                sampling_params,
                result_file,
                gpu_memory_utilization,
                64,  # max_num_seqs (hardcoded for compatibility)
                4096,  # max_model_len (hardcoded for compatibility)
                True,  # enforce_eager (hardcoded for compatibility)
                True,  # trust_remote_code (hardcoded for compatibility)
                enable_timing,
            ),
        )
        proc.start()
        processes.append(proc)
        logger.info(f"Started worker {global_dp_rank} with {len(worker_prompts)} prompts")
    
    # Wait for all processes to complete and collect results
    all_solutions = []
    timeout = 300  # Default timeout of 5 minutes
    
    for i, proc in enumerate(processes):
        proc.join(timeout=timeout)
        if proc.exitcode is None:
            logger.error(f"Worker {i} didn't stop within {timeout} seconds, killing it")
            proc.kill()
        elif proc.exitcode != 0:
            logger.error(f"Worker {i} exited with code {proc.exitcode}")
        else:
            logger.info(f"Worker {i} completed successfully")
            # Read results from file
            try:
                with open(result_files[i], 'r') as f:
                    worker_solutions = json.load(f)
                all_solutions.extend(worker_solutions)
                logger.info(f"Worker {i} contributed {len(worker_solutions)} solutions")
            except Exception as e:
                logger.error(f"Failed to read results from worker {i}: {e}")
    
    # Clean up temporary files
    for result_file in result_files:
        try:
            os.unlink(result_file)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {result_file}: {e}")
    
    # Give engines time to pause their processing loops before exiting
    sleep(1)
    
    logger.info(f"Batch processing completed, generated {len(all_solutions)} solutions")
    return all_solutions

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
    
    # Initialize task queue
    task_queue = TaskQueue(max_tries_per_task=max_tries_per_task)
    
    # Add initial tasks
    setup_start = time.time() if enable_timing else None
    for problem_idx, problem in enumerate(problems):
        problem_text = problem['problem']
        ground_truth = problem['ground_truth']
        
        # Create tasks for both roles
        for role in ['helpful', 'sneaky']:
            for _ in range(k_per_role):
                task_queue.add_task(
                    problem_idx=problem_idx,
                    problem_text=problem_text,
                    ground_truth=ground_truth,
                    role=role
                )
    
    setup_time = time.time() - setup_start if enable_timing else None
    logger.info(f"Added {task_queue.task_counter} initial tasks in {setup_time:.2f}s" if enable_timing else f"Added {task_queue.task_counter} initial tasks")
    
    # Initialize persistent worker pool
    worker_pool = PersistentWorkerPool(
        num_workers=data_parallel_size,
        prover_model_name=prover_model_name,
        gpu_ids=gpu_ids,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_timing=enable_timing
    )
    
    try:
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
                # Add missing fields for task queue processing
                if 'problem_idx' not in solution:
                    # Find the corresponding task
                    for task in batch_tasks:
                        if (task['problem_text'] == solution['problem'] and 
                            task['ground_truth'] == solution['ground_truth'] and 
                            task['role'] == solution['role']):
                            solution['problem_idx'] = task['problem_idx']
                            break
                
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
            logger.info(f"  - Helpful success rate: {status['helpful_success_rate']:.1f}%")
            logger.info(f"  - Sneaky success rate: {status['sneaky_success_rate']:.1f}%")
            if enable_timing:
                logger.info(f"  - Batch time: {iteration_time:.2f}s")
            
    
    finally:
        # Always shutdown workers
        worker_pool.shutdown()
    
    # Final status
    final_status = task_queue.get_status()
    total_time = time.time() - overall_start_time if enable_timing else None
    
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Completed solutions: {final_status['completed_solutions']}")
    logger.info(f"Total attempts: {final_status['total_attempts']}")
    logger.info(f"Helpful success rate: {final_status['helpful_success_rate']:.1f}%")
    logger.info(f"Sneaky success rate: {final_status['sneaky_success_rate']:.1f}%")
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
