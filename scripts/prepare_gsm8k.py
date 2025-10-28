#!/usr/bin/env python3
"""
Prepare GSM8K dataset for training.
"""

import argparse
import json
import os
from typing import List, Dict, Any
import logging
from datasets import load_dataset
import pandas as pd

logger = logging.getLogger(__name__)


def download_gsm8k_dataset() -> Dict[str, List[Dict[str, Any]]]:
    """Download and load the GSM8K dataset."""
    logger.info("Downloading GSM8K dataset...")
    
    # Load dataset from HuggingFace with 'main' config
    dataset = load_dataset("openai/gsm8k", "main")
    
    # Convert to our format
    data = {
        'train': [],
        'val': [],
        'test': []
    }
    
    # Process train data and split into train/val
    train_data = []
    for item in dataset['train']:
        train_data.append({
            'problem': item['question'],
            'solution': item['answer'],
            'ground_truth': extract_final_answer(item['answer'])
        })
    
    # Split train data into train/val (80/20 split)
    import random
    random.seed(42)
    random.shuffle(train_data)
    split_idx = int(0.8 * len(train_data))
    data['train'] = train_data[:split_idx]
    data['val'] = train_data[split_idx:]
    
    # Process test data
    for item in dataset['test']:
        data['test'].append({
            'problem': item['question'],
            'solution': item['answer'],
            'ground_truth': extract_final_answer(item['answer'])
        })
    
    logger.info(f"Loaded {len(data['train'])} training samples, {len(data['val'])} validation samples, and {len(data['test'])} test samples")
    return data

def extract_final_answer(answer: str) -> str:
    """Extract the final answer from GSM8K solution."""
    # GSM8K format: solution ends with "#### <number>"
    import re
    match = re.search(r'####\s*([+-]?\d*\.?\d+)', answer)
    if match:
        return match.group(1)
    return ""

def create_subset(
    data: List[Dict[str, Any]], 
    subset_size: int,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """Create a subset of the data."""
    import random
    random.seed(random_seed)
    
    if subset_size >= len(data):
        return data
    
    return random.sample(data, subset_size)

def save_data(data: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """Save data to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for split, samples in data.items():
        # Save as JSONL
        jsonl_path = os.path.join(output_dir, f"{split}.jsonl")
        with open(jsonl_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        # Save as Parquet
        parquet_path = os.path.join(output_dir, f"{split}.parquet")
        df = pd.DataFrame(samples)
        df.to_parquet(parquet_path, index=False)
        
        logger.info(f"Saved {len(samples)} {split} samples to {jsonl_path} and {parquet_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare GSM8K dataset")
    parser.add_argument("--output-dir", default="data/gsm8k", help="Output directory")
    parser.add_argument("--subset", type=int, help="Create subset of specified size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Download dataset
    data = download_gsm8k_dataset()
    
    # Create subset if requested
    if args.subset:
        logger.info(f"Creating subset of size {args.subset}")
        for split in data:
            data[split] = create_subset(data[split], args.subset, args.seed)
    
    # Save data
    save_data(data, args.output_dir)
    
    logger.info(f"Dataset preparation complete. Data saved to {args.output_dir}")

if __name__ == "__main__":
    main()
