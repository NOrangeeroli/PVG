# dp_infer_mp_fixed.py - Fixed multiprocessing version
import os
import math
import multiprocessing as mp
from time import sleep
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
import tempfile
import json

def worker_proc(
    model_name: str,
    all_prompts: list[str],
    dp_size: int,
    global_dp_rank: int,
    local_dp_rank: int,
    tp_size: int,
    dp_master_ip: str,
    dp_master_port: int,
    result_file: str,
):
    """Worker function that saves results to a file instead of returning them."""
    try:
        # ---- Tell vLLM what DP rank this process is ----
        os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
        os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

        # ---- Shard the global prompt list for this rank ----
        prompts_per_rank = math.ceil(len(all_prompts) / dp_size)
        start = global_dp_rank * prompts_per_rank
        end = min(start + prompts_per_rank, len(all_prompts))
        shard = all_prompts[start:end]
        if len(shard) == 0:
            # Give a dummy prompt so vLLM still initializes cleanly.
            shard = ["Placeholder"]

        print(f"[DP rank {global_dp_rank}] handling {len(shard)} prompts")

        # ---- Per-rank sampling params (can differ if you want) ----
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=64,  # max new tokens to generate
        )

        # ---- Spin up the LLM engine on this rank ----
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,  # GPUs per replica (TP inside each DP rank)
            enforce_eager=True,
            # enable_expert_parallel=True,   # Only for MoE models, not needed for Qwen2.5
        )

        # ---- Run generation for this shard ----
        outputs = llm.generate(shard, sampling_params)

        # ---- Collect results and save to file ----
        summarized = []
        for i, out in enumerate(outputs):
            prompt_text = out.prompt
            gen_text = out.outputs[0].text
            summarized.append(
                {
                    "dp_rank": global_dp_rank,
                    "idx_in_rank": i,
                    "prompt": prompt_text,
                    "generation": gen_text,
                }
            )

        # Save results to file instead of returning
        with open(result_file, 'w') as f:
            json.dump(summarized, f)

        # Let engine flush internal loops before exit
        sleep(1)
        
    except Exception as e:
        print(f"Error in worker {global_dp_rank}: {e}")
        # Save empty results on error
        with open(result_file, 'w') as f:
            json.dump([], f)


def run_dp_inference(
    model_name: str,
    prompts: list[str],
    dp_size: int,
    tp_size: int,
):
    """
    model_name: HF model id or local path
    prompts:    your big batch of requests
    dp_size:    number of data-parallel replicas (usually = num GPUs if tp_size == 1)
    tp_size:    GPUs per replica (tensor parallel group size)
    """

    # For single-node DP we just need one "master ip:port" so the DP ranks
    # can coordinate things like CUDA graph sync / expert-parallel bookkeeping.
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()  # pick a free port

    # Create temporary files for results
    result_files = []
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create result files for each rank
        for rank in range(dp_size):
            result_file = os.path.join(temp_dir, f"results_rank_{rank}.json")
            result_files.append(result_file)

        # Spawn dp_size worker processes.
        # Important: use 'spawn', not 'fork', to avoid CUDA re-init errors.
        mp.set_start_method("spawn", force=True)

        with mp.Pool(processes=dp_size) as pool:
            # Build per-rank args
            jobs = []
            for local_dp_rank, global_dp_rank in enumerate(range(dp_size)):
                jobs.append(
                    (
                        model_name,
                        prompts,
                        dp_size,
                        global_dp_rank,
                        local_dp_rank,
                        tp_size,
                        dp_master_ip,
                        dp_master_port,
                        result_files[global_dp_rank],
                    )
                )

            # Run workers
            pool.starmap(worker_proc, jobs)

        # Collect results from files
        results_per_rank = []
        for result_file in result_files:
            with open(result_file, 'r') as f:
                rank_results = json.load(f)
                results_per_rank.append(rank_results)

    # Flatten results in original-ish order
    merged = [item for rank_list in results_per_rank for item in rank_list]
    return merged


if __name__ == "__main__":
    # EXAMPLE USAGE ---------------------------------
    model = "Qwen/Qwen2.5-0.5B"  # any HF model that fits on a single GPU
    # Batch of requests you want to process offline
    batch_prompts = [
        "Summarize the key ideas of transformers in 2 sentences.",
        "Explain why attention is O(n^2).",
        "Write a haiku about GPUs.",
        "What's the capital of France?",
    ] * 10  # smaller batch for testing

    dp_size = 2   # e.g. 2 GPUs total, 1 replica per GPU
    tp_size = 1   # each replica just uses 1 GPU; no tensor parallel inside

    outputs = run_dp_inference(
        model_name=model,
        prompts=batch_prompts,
        dp_size=dp_size,
        tp_size=tp_size,
    )

    # Print a few merged results
    for row in outputs[:10]:
        print(
            f"[rank {row['dp_rank']}] {row['prompt']!r} -> {row['generation'][:80]!r}"
        )
