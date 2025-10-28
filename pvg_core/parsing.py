"""
Parsing utilities for extracting final answers from model outputs.

Based on the GSM8K format where solutions end with "#### <number>".
"""

import re
from typing import Optional, Union

def extract_final_answer(text: str) -> str:
    """
    Extract the final numerical answer from a model's solution.
    
    Args:
        text: The model's complete solution text
        
    Returns:
        The final answer as a string (normalized)
    """
    # First try to find answer after "####" (GSM8K format)
    gsm8k_match = re.search(r'####\s*([+-]?\d*\.?\d+)', text)
    if gsm8k_match:
        return normalize_number(gsm8k_match.group(1))
    
    # Try to find "Answer: <number>" format
    answer_match = re.search(r'Answer:\s*([+-]?\d*\.?\d+)', text, re.IGNORECASE)
    if answer_match:
        return normalize_number(answer_match.group(1))
    
    # Try to find the last number in the text
    numbers = re.findall(r'([+-]?\d*\.?\d+)', text)
    if numbers:
        return normalize_number(numbers[-1])
    
    # If no number found, return empty string
    return ""

def normalize_number(num_str: str) -> str:
    """
    Normalize a number string to a standard format.
    
    Args:
        num_str: String representation of a number
        
    Returns:
        Normalized number string
    """
    # Remove leading/trailing whitespace
    num_str = num_str.strip()
    
    # Handle empty string
    if not num_str:
        return ""
    
    # Try to convert to float to normalize
    try:
        num = float(num_str)
        # If it's an integer, return without decimal point
        if num.is_integer():
            return str(int(num))
        else:
            return str(num)
    except ValueError:
        # If conversion fails, return the original string
        return num_str

def is_correct(predicted_answer: str, ground_truth_answer: str) -> bool:
    """
    Check if a predicted answer matches the ground truth.
    
    Args:
        predicted_answer: The model's predicted answer
        ground_truth_answer: The correct answer
        
    Returns:
        True if the answers match (after normalization)
    """
    pred_norm = normalize_number(predicted_answer)
    gt_norm = normalize_number(ground_truth_answer)
    
    return pred_norm == gt_norm

def extract_solution_steps(text: str) -> list[str]:
    """
    Extract individual solution steps from a model's output.
    
    Args:
        text: The model's complete solution text
        
    Returns:
        List of solution steps
    """
    # Split by common step indicators
    step_patterns = [
        r'Step \d+:',
        r'\d+\.',
        r'First,',
        r'Next,',
        r'Then,',
        r'Finally,',
        r'Therefore,',
        r'Hence,',
    ]
    
    steps = []
    current_step = ""
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line starts a new step
        is_new_step = any(re.match(pattern, line, re.IGNORECASE) for pattern in step_patterns)
        
        if is_new_step and current_step:
            steps.append(current_step.strip())
            current_step = line
        else:
            current_step += " " + line if current_step else line
    
    # Add the last step
    if current_step:
        steps.append(current_step.strip())
    
    return steps

def extract_reasoning_quality_indicators(text: str) -> dict[str, bool]:
    """
    Extract indicators of reasoning quality from a solution.
    
    Args:
        text: The model's complete solution text
        
    Returns:
        Dictionary of quality indicators
    """
    indicators = {
        'has_step_by_step': bool(re.search(r'(step|first|next|then|finally)', text, re.IGNORECASE)),
        'has_calculations': bool(re.search(r'[\+\-\*/=]', text)),
        'has_explanations': bool(re.search(r'(because|since|therefore|hence|so)', text, re.IGNORECASE)),
        'has_units': bool(re.search(r'\b(years?|months?|days?|hours?|minutes?|seconds?|dollars?|cents?|pounds?|kg|grams?|meters?|cm|inches?|feet?)\b', text, re.IGNORECASE)),
        'has_final_answer': bool(re.search(r'(answer|####)', text, re.IGNORECASE)),
        'is_concise': len(text.split()) < 200,  # Less than 200 words
        'has_clear_structure': bool(re.search(r'(#|step|first|next|then|finally)', text, re.IGNORECASE)),
    }
    
    return indicators
