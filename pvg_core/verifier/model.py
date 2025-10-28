"""
Verifier model implementation with classification head.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from transformers import AutoModel, AutoTokenizer, AutoConfig
import logging

logger = logging.getLogger(__name__)

class VerifierModel(nn.Module):
    """
    Verifier model that takes problem-solution pairs and outputs correctness scores.
    """
    
    def __init__(
        self,
        base_model_name: str,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        freeze_base: bool = False,
        **kwargs
    ):
        """
        Initialize the verifier model.
        
        Args:
            base_model_name: HuggingFace model name for the base LM
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Dropout rate for the classification head
            freeze_base: Whether to freeze the base model weights
            **kwargs: Additional arguments for model loading
        """
        super().__init__()
        
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.freeze_base = freeze_base
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(
            base_model_name,
            **kwargs
        )
        
        # Get hidden size
        self.hidden_size = self.base_model.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the verifier model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            labels: Ground truth labels of shape (batch_size,)
            
        Returns:
            Dictionary containing logits, loss, and predictions
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token or mean pooling
        if hasattr(self.base_model.config, 'pooler_output') and self.base_model.config.pooler_output:
            # Use pooler output if available
            pooled_output = outputs.pooler_output
        else:
            # Use mean pooling over sequence length
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                # Simple mean pooling
                pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'loss': loss,
            'predictions': predictions,
            'hidden_states': outputs.last_hidden_state,
            'pooled_output': pooled_output
        }
    
    def get_verifier_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get verifier scores (probability of correctness) for input pairs.
        
        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            
        Returns:
            Verifier scores of shape (batch_size,) in range [0, 1]
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            # Get probability of "correct" class (assuming class 1 is correct)
            probs = torch.softmax(outputs['logits'], dim=-1)
            scores = probs[:, 1]  # Probability of correct class
            return scores
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model to a directory."""
        self.base_model.save_pretrained(save_directory, **kwargs)
        
        # Save classifier weights separately
        classifier_state = {
            'classifier.weight': self.classifier.weight.state_dict(),
            'classifier.bias': self.classifier.bias.state_dict(),
            'dropout.p': self.dropout.p,
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_size
        }
        
        torch.save(classifier_state, f"{save_directory}/classifier.pt")
    
    @classmethod
    def from_pretrained(cls, model_directory: str, **kwargs):
        """Load a pretrained verifier model."""
        # Load base model
        base_model = AutoModel.from_pretrained(model_directory, **kwargs)
        
        # Load classifier weights
        classifier_state = torch.load(f"{model_directory}/classifier.pt")
        
        # Create model instance
        model = cls(
            base_model_name=model_directory,
            num_classes=classifier_state['num_classes'],
            dropout_rate=classifier_state['dropout.p'],
            **kwargs
        )
        
        # Load classifier weights
        model.classifier.load_state_dict(classifier_state['classifier.weight'])
        model.classifier.bias.load_state_dict(classifier_state['classifier.bias'])
        
        return model

class VerifierTokenizer:
    """Tokenizer wrapper for verifier inputs."""
    
    def __init__(self, tokenizer_name: str, max_length: int = 512):
        """
        Initialize the verifier tokenizer.
        
        Args:
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode_pair(
        self,
        problem: str,
        solution: str,
        truncation: bool = True,
        padding: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a problem-solution pair for the verifier.
        
        Args:
            problem: The math problem text
            solution: The solution text
            truncation: Whether to truncate long sequences
            padding: Whether to pad sequences
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Format the input
        text = f"Problem: {problem}\nSolution: {solution}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=truncation,
            padding=padding,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
    
    def encode_batch(
        self,
        problems: list[str],
        solutions: list[str],
        truncation: bool = True,
        padding: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of problem-solution pairs.
        
        Args:
            problems: List of problem texts
            solutions: List of solution texts
            truncation: Whether to truncate long sequences
            padding: Whether to pad sequences
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Format inputs
        texts = [f"Problem: {p}\nSolution: {s}" for p, s in zip(problems, solutions)]
        
        # Tokenize batch
        encoding = self.tokenizer(
            texts,
            truncation=truncation,
            padding=padding,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
