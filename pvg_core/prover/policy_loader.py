"""
Policy loader for loading and managing prover models with PEFT/LoRA support.
"""

import torch
from typing import Optional, Dict, Any, Union
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel,
    PeftConfig
)
import logging

logger = logging.getLogger(__name__)

class ProverPolicyLoader:
    """Loader for prover models with PEFT support."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        use_peft: bool = True,
        peft_config: Optional[Dict[str, Any]] = None,
        use_quantization: bool = False,
        quantization_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the prover policy loader.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on
            torch_dtype: Data type for model weights
            use_peft: Whether to use PEFT/LoRA
            peft_config: Configuration for PEFT
            use_quantization: Whether to use quantization
            quantization_config: Configuration for quantization
            **kwargs: Additional arguments for model loading
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_peft = use_peft
        self.use_quantization = use_quantization
        
        # Default PEFT config
        if peft_config is None:
            peft_config = {
                'r': 16,
                'lora_alpha': 32,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                'lora_dropout': 0.1,
                'bias': 'none',
                'task_type': TaskType.CAUSAL_LM
            }
        self.peft_config = peft_config
        
        # Default quantization config
        if quantization_config is None:
            quantization_config = {
                'load_in_4bit': True,
                'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_compute_dtype': torch.float16,
                'bnb_4bit_use_double_quant': True
            }
        self.quantization_config = quantization_config
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self._load_model(**kwargs)
    
    def _load_model(self, **kwargs):
        """Load the base model with optional quantization."""
        model_kwargs = {
            'torch_dtype': self.torch_dtype,
            'device_map': self.device,
            **kwargs
        }
        
        if self.use_quantization:
            # Setup quantization config
            bnb_config = BitsAndBytesConfig(**self.quantization_config)
            model_kwargs['quantization_config'] = bnb_config
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Apply PEFT if requested
        if self.use_peft:
            self._apply_peft()
        else:
            self.model = self.base_model
    
    def _apply_peft(self):
        """Apply PEFT/LoRA to the model."""
        # Create LoRA config
        lora_config = LoraConfig(**self.peft_config)
        
        # Apply PEFT
        self.model = get_peft_model(self.base_model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("Applied PEFT/LoRA to the model")
    
    def get_model(self):
        """Get the loaded model."""
        return self.model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer
    
    def get_base_model(self):
        """Get the base model (without PEFT)."""
        return self.base_model
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model and tokenizer."""
        # Save PEFT model
        if self.use_peft:
            self.model.save_pretrained(save_directory, **kwargs)
        else:
            self.model.save_pretrained(save_directory, **kwargs)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        logger.info(f"Model saved to {save_directory}")
    
    def load_pretrained(self, model_directory: str, **kwargs):
        """Load a pretrained model."""
        if self.use_peft:
            # Load PEFT model
            self.model = PeftModel.from_pretrained(
                self.base_model,
                model_directory,
                **kwargs
            )
        else:
            # Load regular model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_directory,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                **kwargs
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        
        logger.info(f"Model loaded from {model_directory}")
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get information about trainable parameters."""
        if self.use_peft:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.model.parameters())
            return {
                'trainable': trainable_params,
                'total': all_params,
                'percentage': 100 * trainable_params / all_params
            }
        else:
            all_params = sum(p.numel() for p in self.model.parameters())
            return {
                'trainable': all_params,
                'total': all_params,
                'percentage': 100.0
            }
    
    def freeze_base_model(self):
        """Freeze the base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        logger.info("Frozen base model parameters")
    
    def unfreeze_base_model(self):
        """Unfreeze the base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = True
        
        logger.info("Unfrozen base model parameters")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'torch_dtype': str(self.torch_dtype),
            'use_peft': self.use_peft,
            'use_quantization': self.use_quantization,
            'peft_config': self.peft_config if self.use_peft else None,
            'quantization_config': self.quantization_config if self.use_quantization else None,
            'parameters': self.get_trainable_parameters()
        }

class ReferenceModelLoader:
    """Loader for reference models used in PPO training."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        """
        Initialize the reference model loader.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on
            torch_dtype: Data type for model weights
            **kwargs: Additional arguments for model loading
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load model (no PEFT for reference)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            **kwargs
        )
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        logger.info(f"Loaded reference model: {model_name}")
    
    def get_model(self):
        """Get the reference model."""
        return self.model
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the reference model."""
        self.model.save_pretrained(save_directory, **kwargs)
        logger.info(f"Reference model saved to {save_directory}")
    
    def load_pretrained(self, model_directory: str, **kwargs):
        """Load a pretrained reference model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_directory,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            **kwargs
        )
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        logger.info(f"Reference model loaded from {model_directory}")
