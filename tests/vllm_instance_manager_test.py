# Test our VLLMInstanceManager for data parallelism
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pvg_core.verifier.fast_sampling import VLLMInstanceManager
from vllm import SamplingParams
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vllm_instance_manager():
    """Test the VLLMInstanceManager with multiple GPUs."""
    
    # Configuration
    model_name = "Qwen/Qwen2.5-0.5B"
    gpu_ids = [2, 3, 4, 5]  # Use 4 GPUs
    gpu_memory_utilization = 0.1  # Conservative memory usage
    
    # Test prompts
    test_prompts = [
        "What is 2+2?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about machine learning.",
        "What is the capital of Japan?",
        "Explain the concept of attention in transformers.",
        "What is the speed of light?",
        "Write a short poem about GPUs.",
        "What is the largest planet in our solar system?",
    ]
    
    # Initialize the manager
    logger.info("🚀 Initializing VLLMInstanceManager...")
    manager = VLLMInstanceManager(
        model_name=model_name,
        gpu_ids=gpu_ids,
        gpu_memory_utilization=gpu_memory_utilization
    )
    
    try:
        # Initialize all instances
        manager.initialize_instances()
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=64,
        )
        
        # Test parallel generation
        logger.info("🔄 Testing parallel generation...")
        outputs = manager.generate_parallel(test_prompts, sampling_params)
        
        # Display results
        logger.info("📊 Results:")
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generation = output.outputs[0].text
            logger.info(f"  {i+1}. {prompt} -> {generation[:80]}...")
        
        logger.info(f"✅ Successfully generated {len(outputs)} outputs using {len(manager.instances)} VLLM instances")
        
    except Exception as e:
        logger.error(f"❌ Error during testing: {e}")
        raise
    finally:
        # Clean up
        logger.info("🧹 Cleaning up...")

if __name__ == "__main__":
    test_vllm_instance_manager()

