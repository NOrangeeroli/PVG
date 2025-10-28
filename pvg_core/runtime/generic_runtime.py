import os
import logging
import json
import time
import queue
from typing import List, Dict, Any, Optional, Callable
from multiprocessing import Process, set_start_method, Queue, Event

from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


# Ensure CUDA-safe start method for multiprocessing
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass


class TaskQueue:
    """
    Generic task queue that tracks tries per task and delegates success logic
    to a provided callback. The queue does not assume task schema beyond
    requiring a stable 'task_id' assigned on add_task.
    """

    def __init__(self, is_success_fn: Callable[[Dict[str, Any]], bool], max_tries_per_task: int = 5):
        self.max_tries_per_task = max_tries_per_task
        self.is_success_fn = is_success_fn
        self.pending_tasks: List[Dict[str, Any]] = []
        self.completed_solutions: List[Dict[str, Any]] = []
        self.failed_tasks: List[Dict[str, Any]] = []
        self.task_counter = 0
        # Registry of original tasks by id to allow resubmission and track tries
        self.task_registry: Dict[str, Dict[str, Any]] = {}

    def add_task(self, task: Dict[str, Any]):
        """Add a new (generic) task to the queue. Assigns task_id if missing."""
        import uuid
        task_id = task.get('task_id', str(uuid.uuid4()))
        task['task_id'] = task_id
        task.setdefault('tries', 0)
        self.pending_tasks.append(task)
        self.task_registry[task_id] = task
        self.task_counter += 1

    def is_solution_successful(self, solution: Dict[str, Any]) -> bool:
        """Determine if a solution is successful via provided callback."""
        return self.is_success_fn(solution)

    def process_solution(self, solution: Dict[str, Any]) -> bool:
        """Process a solution, update tries, and enqueue for resampling if needed."""
        is_successful = self.is_solution_successful(solution)

        # Track tries using task_registry
        task_id = solution.get('task_id')
        assert task_id is not None
        assert task_id in self.task_registry

        self.task_registry[task_id]['tries'] += 1
        current_tries = self.task_registry[task_id]['tries']

        if is_successful:
            self.completed_solutions.append(solution)
            logger.info(f"Successful task added (total: {len(self.completed_solutions)})")
            return True
        else:
            # Check if task has reached max tries
            if current_tries >= self.max_tries_per_task:
                logger.info(f"Task {task_id} reached max tries ({self.max_tries_per_task}), removing from queue")
                return False

            # Add failed task back to queue for resampling
            base_task = self.task_registry.get(task_id, {})
            failed_task = dict(base_task)
            failed_task['tries'] = current_tries
            self.failed_tasks.append(failed_task)
            logger.info(f"Failed task {task_id} added to resample queue (tries: {current_tries}/{self.max_tries_per_task})")
            return False

    def get_next_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get next batch of tasks for processing."""
        self.pending_tasks.extend(self.failed_tasks)
        self.failed_tasks.clear()
        batch = self.pending_tasks[:batch_size]
        self.pending_tasks = self.pending_tasks[batch_size:]
        return batch

    def should_continue(self) -> bool:
        if len(self.pending_tasks) == 0 and len(self.failed_tasks) == 0:
            logger.warning("No more tasks to process, stopping")
            return False
        return True

    def get_status(self) -> Dict[str, Any]:
        total_attempts = sum(task.get('tries', 0) for task in self.task_registry.values()) if self.task_registry else 0
        return {
            'completed_solutions': len(self.completed_solutions),
            'pending_tasks': len(self.pending_tasks),
            'failed_tasks': len(self.failed_tasks),
            'max_tries_per_task': self.max_tries_per_task,
            'total_attempts': total_attempts
        }

    def generate_report(self, output_dir: str = ".") -> str:
        """Generate report by dumping the task_registry mapping as JSON"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"task_queue_report_{timestamp}.json"
        report_path = os.path.join(output_dir, report_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.task_registry, f, indent=2)
        logger.info(f"Task queue report saved to: {report_path}")
        return report_path


class PersistentWorkerPool:
    """
    Generic pool of persistent VLLM workers that process batches of tasks.
    Each task must contain a 'prompt' field used for generation. The worker
    copies task metadata and adds 'solution' and 'worker_id' to results.
    """

    def __init__(
        self,
        num_workers: int,
        model_name: str,
        gpu_ids: List[int],
        gpu_memory_utilization: float = 0.05,
        max_num_seqs: int = 64,
        max_model_len: int = 4096,
        enable_timing: bool = True,
        worker_batch_size: int = 8,
        show_progress: bool = True,
    ):
        self.num_workers = num_workers
        self.model_name = model_name
        self.gpu_ids = gpu_ids
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.enable_timing = enable_timing
        self.worker_batch_size = worker_batch_size
        self.show_progress = show_progress

        # Communication queues
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.shutdown_event = Event()

        # Worker processes
        self.workers: List[Process] = []

        # Statistics
        self.total_tasks_processed = 0
        self.total_results_collected = 0

        logger.info(f"Initializing PersistentWorkerPool with {num_workers} workers")

    def start_workers(self):
        logger.info("Starting persistent VLLM workers...")
        for worker_id in range(self.num_workers):
            gpu_id = self.gpu_ids[worker_id] if worker_id < len(self.gpu_ids) else worker_id
            worker_process = Process(
                target=self._persistent_worker,
                args=(
                    worker_id,
                    gpu_id,
                    self.model_name,
                    self.gpu_memory_utilization,
                    self.max_num_seqs,
                    self.max_model_len,
                    self.enable_timing,
                    self.show_progress,
                    self.task_queue,
                    self.result_queue,
                    self.shutdown_event,
                ),
                name=f"PersistentWorker-{worker_id}",
            )
            worker_process.start()
            self.workers.append(worker_process)
            logger.info(f"Started persistent worker {worker_id} on GPU {gpu_id}")
        time.sleep(5)
        logger.info("All persistent workers started and initialized")

    def submit_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        logger.info(f"Submitting {len(tasks)} tasks to worker pool (batch size per worker: {self.worker_batch_size})")
        for i in range(0, len(tasks), self.worker_batch_size):
            batch = tasks[i:i + self.worker_batch_size]
            self.task_queue.put(batch)
            self.total_tasks_processed += len(batch)

    def collect_results(self, timeout: float = 30.0) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result = self.result_queue.get(timeout=0.1)
                results.append(result)
                self.total_results_collected += 1
            except queue.Empty:
                break
        if results:
            logger.info(f"Collected {len(results)} results from worker pool")
        return results

    def wait_for_completion(self, expected_results: int, timeout: float = 300.0) -> List[Dict[str, Any]]:
        logger.info(f"Waiting for {expected_results} results to complete...")
        all_results: List[Dict[str, Any]] = []
        start_time = time.time()
        while len(all_results) < expected_results and time.time() - start_time < timeout:
            batch_results = self.collect_results(timeout=1.0)
            all_results.extend(batch_results)
            if len(all_results) < expected_results:
                logger.info(f"Progress: {len(all_results)}/{expected_results} results completed")
                time.sleep(0.5)
        if len(all_results) >= expected_results:
            logger.info(f"Successfully collected all {len(all_results)} expected results")
        else:
            logger.warning(f"Timeout reached. Collected {len(all_results)}/{expected_results} results")
        return all_results

    def shutdown(self):
        logger.info("Shutting down persistent worker pool...")
        self.shutdown_event.set()
        for i, worker in enumerate(self.workers):
            logger.info(f"Waiting for worker {i} to finish...")
            worker.join(timeout=30)
            if worker.is_alive():
                logger.warning(f"Worker {i} didn't stop gracefully, terminating...")
                worker.terminate()
                worker.join()
        logger.info("All workers shut down")

    @staticmethod
    def _persistent_worker(
        worker_id: int,
        gpu_id: int,
        model_name: str,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        max_model_len: int,
        enable_timing: bool,
        show_progress: bool,
        task_queue: Queue,
        result_queue: Queue,
        shutdown_event: Event,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if not show_progress:
            # Disable tqdm progress bars inside this worker process
            os.environ["TQDM_DISABLE"] = "1"
        logger.info(f"Persistent worker {worker_id} starting on GPU {gpu_id}")

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

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop=["\n\n\n"],
        )

        logger.info(f"Persistent worker {worker_id} ready and waiting for tasks")
        tasks_processed = 0
        while not shutdown_event.is_set():
            try:
                tasks_batch = task_queue.get(timeout=1.0)
                if not isinstance(tasks_batch, list):
                    tasks_batch = [tasks_batch]

                logger.info(f"Worker {worker_id} processing batch of {len(tasks_batch)} tasks")
                prompts = [t['prompt'] for t in tasks_batch]

                gen_start = time.time() if enable_timing else None
                outputs = llm.generate(prompts, sampling_params)
                gen_time = time.time() - gen_start if enable_timing else None

                if outputs and len(outputs) > 0:
                    for t, out in zip(tasks_batch, outputs):
                        generated_text = out.outputs[0].text
                        result = dict(t)
                        result.update({
                            'solution': generated_text,
                            'worker_id': worker_id,
                            'task_id': t.get('task_id'),
                        })
                        if enable_timing:
                            result.update({
                                'generation_time': gen_time,
                                'tokens_generated': len(generated_text.split()),
                            })
                        result_queue.put(result)
                        tasks_processed += 1
                    logger.info(f"Worker {worker_id} completed batch (total tasks processed: {tasks_processed})")
                else:
                    logger.error(f"Worker {worker_id} got empty outputs for batch of size {len(tasks_batch)}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error processing batch: {e}")
                continue

        logger.info(f"Persistent worker {worker_id} shutting down after processing {tasks_processed} tasks")


