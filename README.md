# PVG (Prover-Verifier Game) Training System

A multi-GPU training system for Prover-Verifier Game scenarios using VLLM for fast sampling and data parallelism.

## Features

- **Multi-GPU Support**: Efficient data parallel processing using VLLM
- **Task Queue System**: Role-specific success criteria with retry mechanisms
- **Persistent Worker Pool**: Optimized GPU utilization with persistent VLLM workers
- **Comprehensive Reporting**: Detailed task attempt tracking and performance metrics
- **Configurable Sampling**: Flexible configuration for different training scenarios

## Architecture

### Core Components

- **Fast Sampling**: VLLM-based high-throughput solution generation
- **Task Queue**: Manages sampling tasks with role-specific success criteria
  - Helpful Prover: Success = Correct solution, Failure = Wrong solution → Resample
  - Sneaky Prover: Success = Wrong solution, Failure = Correct solution → Resample
- **Persistent Worker Pool**: Maintains VLLM instances across batches for efficiency
- **Multi-GPU Data Parallelism**: Distributes workload across multiple GPUs

### Key Features

- **Per-task Retry Limits**: Configurable maximum attempts per task
- **Detailed Reporting**: Comprehensive task attempt tracking and success analysis
- **Memory Optimization**: Conservative GPU memory utilization
- **Error Handling**: Robust error handling and fallback mechanisms

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd PVG
```

2. Create a virtual environment:
```bash
conda create -n pvg python=3.10
conda activate pvg
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Multi-GPU Configuration (`configs/multi_gpu.yaml`)

```yaml
verifier:
  sampling:
    k_per_role: 2  # Solutions per role per problem
    max_tries_per_task: 5  # Maximum attempts per task
    data_parallel_size: 3  # Number of GPU workers
    gpu_memory_utilization: 0.05  # Conservative memory usage
    gpu_ids: [2, 3, 4]  # Specific GPUs to use
```

### Default Configuration (`configs/default.yaml`)

```yaml
verifier:
  sampling:
    k_per_role: 5
    max_tries_per_task: 10
    data_parallel_size: null  # Single GPU
    gpu_memory_utilization: 0.8
```

## Usage

### Basic Training

```bash
python scripts/run_round.py \
  --config configs/multi_gpu.yaml \
  --data-dir data/gsm8k \
  --output-dir outputs/round_001 \
  --multi-gpu
```

### Single GPU Training

```bash
python scripts/run_round.py \
  --config configs/default.yaml \
  --data-dir data/gsm8k \
  --output-dir outputs/round_001
```

### Testing Fast Sampling

```bash
python -c "from pvg_core.verifier.fast_sampling import example_usage; example_usage()"
```

## Project Structure

```
PVG/
├── configs/                 # Configuration files
│   ├── default.yaml        # Single GPU configuration
│   └── multi_gpu.yaml      # Multi-GPU configuration
├── pvg_core/               # Core implementation
│   ├── verifier/           # Verifier components
│   │   ├── fast_sampling.py # VLLM-based fast sampling
│   │   └── sample_solutions.py # Solution sampling logic
│   ├── parsing/            # Answer parsing utilities
│   └── prompts/            # Prompt formatting
├── scripts/                # Training scripts
│   └── run_round.py        # Main training script
├── data/                   # Data directory (not in git)
├── outputs/                # Output directory (not in git)
└── requirements.txt        # Python dependencies
```

## Key Classes

### TaskQueue
- Manages sampling tasks with role-specific success criteria
- Tracks retry attempts and generates detailed reports
- Handles task lifecycle from creation to completion

### PersistentWorkerPool
- Maintains VLLM worker processes across batches
- Optimizes GPU utilization and reduces initialization overhead
- Handles inter-process communication and result collection

### FastSampler
- VLLM-based high-throughput solution generation
- Supports data parallelism and multi-GPU processing
- Includes comprehensive error handling and fallback mechanisms

## Performance Features

- **Memory Optimization**: Conservative GPU memory utilization (0.05 by default)
- **Batch Processing**: Configurable batch sizes for optimal throughput
- **Persistent Workers**: Eliminates VLLM reinitialization overhead
- **Dynamic Port Allocation**: Avoids port conflicts in multi-process setups
- **Comprehensive Logging**: Detailed performance metrics and debugging information

## Error Handling

- **GPU Memory Management**: Automatic memory utilization adjustment
- **Process Cleanup**: Graceful shutdown of worker processes
- **Fallback Mechanisms**: Automatic fallback to standard sampling if VLLM fails
- **Retry Logic**: Configurable retry limits with detailed attempt tracking

## Reporting

The system generates detailed reports including:
- Task success/failure statistics
- Role-specific performance metrics
- Individual task attempt histories
- Performance timing information
- Resource utilization statistics

## Requirements

- Python 3.10+
- CUDA-compatible GPUs
- VLLM library
- PyTorch
- Transformers
- Other dependencies listed in `requirements.txt`

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Citation

[Add citation information if applicable]