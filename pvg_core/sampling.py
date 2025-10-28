"""
Batched generation utilities for sampling from language models.
"""

import torch
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import logging

logger = logging.getLogger(__name__)

class ModelSampler:
    """Utility class for sampling from language models."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        use_cache: bool = True,
        **kwargs
    ):
        """
        Initialize the model sampler.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on
            torch_dtype: Data type for model weights
            use_cache: Whether to use KV cache
            **kwargs: Additional arguments for model loading
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            use_cache=use_cache,
            **kwargs
        )
        
        # Set generation config
        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512,
            **kwargs.get('generation_config', {})
        )
        
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate per prompt
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Update generation config
        generation_config = GenerationConfig(
            **self.generation_config.to_dict(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            **kwargs
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode outputs
        responses = []
        for i, output in enumerate(outputs):
            # Remove input tokens
            input_length = inputs['input_ids'][i // num_return_sequences].shape[0]
            generated_tokens = output[input_length:]
            
            # Decode
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def generate_with_roles(
        self,
        problems: List[str],
        roles: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate solutions with role conditioning.
        
        Args:
            problems: List of math problems
            roles: List of roles ("helpful" or "sneaky")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List of dictionaries with problem, role, solution, and metadata
        """
        from .prompts import format_simple_prompt
        
        # Format prompts with roles
        prompts = []
        for problem, role in zip(problems, roles):
            prompt = format_simple_prompt(problem, role)
            prompts.append(prompt)
        
        # Generate solutions
        solutions = self.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # Package results
        results = []
        for problem, role, solution in zip(problems, roles, solutions):
            results.append({
                'problem': problem,
                'role': role,
                'solution': solution,
                'model_name': self.model_name,
                'generation_config': {
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    **kwargs
                }
            })
        
        return results
    
    def generate_verifier_scores(
        self,
        problem_solution_pairs: List[tuple[str, str]],
        max_new_tokens: int = 10,
        temperature: float = 0.1,
        **kwargs
    ) -> List[float]:
        """
        Generate verifier scores for problem-solution pairs.
        
        Args:
            problem_solution_pairs: List of (problem, solution) tuples
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List of verifier scores (0.0 to 1.0)
        """
        from .prompts import get_verifier_classification_prompt
        
        # Format prompts
        prompts = []
        for problem, solution in problem_solution_pairs:
            prompt = get_verifier_classification_prompt(problem, solution)
            prompts.append(prompt)
        
        # Generate classifications
        responses = self.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # Convert to scores
        scores = []
        for response in responses:
            response_lower = response.lower().strip()
            if 'correct' in response_lower and 'incorrect' not in response_lower:
                scores.append(1.0)
            elif 'incorrect' in response_lower:
                scores.append(0.0)
            else:
                # Default to 0.5 if unclear
                scores.append(0.5)
        
        return scores

def sample_with_roles(
    model_name: str,
    problems: List[str],
    roles: List[str],
    device: str = "auto",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function to sample solutions with roles.
    
    Args:
        model_name: HuggingFace model name
        problems: List of math problems
        roles: List of roles
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        List of generated solutions with metadata
    """
    sampler = ModelSampler(model_name, device=device, **kwargs)
    return sampler.generate_with_roles(problems, roles, **kwargs)
