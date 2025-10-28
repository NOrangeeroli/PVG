"""
Training loop for the verifier model using supervised learning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm
import json
import os

from .model import VerifierModel, VerifierTokenizer

logger = logging.getLogger(__name__)

class VerifierDataset(Dataset):
    """Dataset for verifier training."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: VerifierTokenizer,
        max_length: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            data: List of dictionaries with keys: problem, solution, is_correct
            tokenizer: Verifier tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode the problem-solution pair
        encoding = self.tokenizer.encode_pair(
            item['problem'],
            item['solution'],
            truncation=True,
            padding=False
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(int(item['is_correct']), dtype=torch.long),  # Convert bool to int
            'problem': item['problem'],
            'solution': item['solution']
        }

class VerifierTrainer:
    """Trainer for the verifier model."""
    
    def __init__(
        self,
        model: VerifierModel,
        tokenizer: VerifierTokenizer,
        device: str = "cuda",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        use_multi_gpu: bool = False,
        **kwargs
    ):
        """
        Initialize the verifier trainer.
        
        Args:
            model: Verifier model
            tokenizer: Verifier tokenizer
            device: Device to use for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps
            **kwargs: Additional arguments
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.use_multi_gpu = use_multi_gpu
        
        # Setup multi-GPU if enabled
        if use_multi_gpu and torch.cuda.device_count() > 1:
            logger.info("Using %d GPUs for verifier training", torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_history = []
        self.val_history = []
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        max_grad_norm: float = 1.0
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            predictions = outputs['predictions']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{total_correct / total_samples:.4f}"
            })
        
        # Calculate metrics
        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(dataloader)
        
        # Calculate AUC
        try:
            auc_score = roc_auc_score(all_labels, all_predictions)
        except ValueError:
            auc_score = 0.5  # Default for binary classification
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc_score
        }
        
        self.train_history.append(metrics)
        return metrics
    
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                predictions = outputs['predictions']
                logits = outputs['logits']
                
                # Get verifier scores
                scores = torch.softmax(logits, dim=-1)[:, 1]  # Probability of correct class
                
                # Update metrics
                total_loss += loss.item()
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
        
        # Calculate metrics
        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(dataloader)
        
        # Calculate AUC
        try:
            auc_score = roc_auc_score(all_labels, all_scores)
        except ValueError:
            auc_score = 0.5
        
        # Calculate PR-AUC
        try:
            precision, recall, _ = precision_recall_curve(all_labels, all_scores)
            pr_auc = auc(recall, precision)
        except ValueError:
            pr_auc = 0.5
        
        # Calculate ECE (Expected Calibration Error)
        ece = self._calculate_ece(all_scores, all_labels)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc_score,
            'pr_auc': pr_auc,
            'ece': ece
        }
        
        self.val_history.append(metrics)
        return metrics
    
    def _calculate_ece(
        self,
        scores: List[float],
        labels: List[int],
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            scores: List of confidence scores
            labels: List of true labels
            n_bins: Number of bins for calibration
            
        Returns:
            ECE value
        """
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (scores > bin_lower) & (scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence in this bin
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = scores[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 3,
        save_every: int = 1,
        save_dir: Optional[str] = None,
        early_stopping_patience: int = 3
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Train the verifier model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            save_every: Save model every N epochs
            save_dir: Directory to save models
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary with training history
        """
        best_val_auc = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader, epoch)
            logger.info(f"Train metrics: {train_metrics}")
            
            # Evaluate
            val_metrics = self.evaluate(val_dataloader)
            logger.info(f"Validation metrics: {val_metrics}")
            
            # Save model if best
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                
                if save_dir:
                    self.save_model(save_dir, epoch)
                    logger.info(f"Saved best model with AUC: {best_val_auc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        return {
            'train': self.train_history,
            'val': self.val_history
        }
    
    def save_model(self, save_dir: str, epoch: int):
        """Save the model and tokenizer."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_dir)
        
        # Save tokenizer
        self.tokenizer.tokenizer.save_pretrained(save_dir)
        
        # Save training history
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'epoch': epoch
        }
        
        with open(f"{save_dir}/training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Model saved to {save_dir}")
    
    def load_model(self, model_dir: str):
        """Load a pretrained model."""
        self.model = VerifierModel.from_pretrained(model_dir)
        self.model.to(self.device)
        
        # Load training history if available
        history_file = f"{model_dir}/training_history.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
                self.train_history = history.get('train', [])
                self.val_history = history.get('val', [])
        
        logger.info(f"Model loaded from {model_dir}")
