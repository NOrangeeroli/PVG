"""
Human proxy evaluation for measuring legibility to time-constrained humans.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class HumanEvaluationResult:
    """Result of human evaluation."""
    problem: str
    solution: str
    ground_truth: str
    human_judgment: bool
    response_time: float
    confidence: float
    evaluator_id: str
    timestamp: str

class HumanProxyEvaluator:
    """Evaluator that simulates human evaluation with time constraints."""
    
    def __init__(
        self,
        judge_model,
        tokenizer,
        time_limit: float = 45.0,  # seconds
        device: str = "cuda"
    ):
        """
        Initialize the human proxy evaluator.
        
        Args:
            judge_model: Small language model to act as human judge
            tokenizer: Tokenizer for the judge model
            time_limit: Time limit for evaluation (seconds)
            device: Device to use
        """
        self.judge_model = judge_model
        self.tokenizer = tokenizer
        self.time_limit = time_limit
        self.device = device
    
    def evaluate_solution(
        self,
        problem: str,
        solution: str,
        ground_truth: str,
        evaluator_id: str = "proxy"
    ) -> HumanEvaluationResult:
        """
        Evaluate a single solution with time constraints.
        
        Args:
            problem: Math problem
            solution: Generated solution
            ground_truth: Ground truth answer
            evaluator_id: ID of the evaluator
            
        Returns:
            Human evaluation result
        """
        start_time = datetime.now()
        
        # Format the evaluation prompt
        prompt = self._format_evaluation_prompt(problem, solution)
        
        # Tokenize and generate judgment
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate judgment with time constraint
        with torch.no_grad():
            outputs = self.judge_model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode judgment
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        judgment_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Parse judgment
        human_judgment = self._parse_judgment(judgment_text)
        
        # Calculate response time
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        # Calculate confidence (based on response time and judgment clarity)
        confidence = self._calculate_confidence(response_time, judgment_text)
        
        return HumanEvaluationResult(
            problem=problem,
            solution=solution,
            ground_truth=ground_truth,
            human_judgment=human_judgment,
            response_time=response_time,
            confidence=confidence,
            evaluator_id=evaluator_id,
            timestamp=start_time.isoformat()
        )
    
    def _format_evaluation_prompt(self, problem: str, solution: str) -> str:
        """Format the evaluation prompt for the human judge."""
        return f"""You are evaluating a math solution. You have {self.time_limit} seconds to determine if the solution is correct.

Problem: {problem}

Solution: {solution}

Is this solution correct? Answer with "Correct" or "Incorrect"."""
    
    def _parse_judgment(self, judgment_text: str) -> bool:
        """Parse the judgment text to get a boolean result."""
        judgment_lower = judgment_text.lower().strip()
        
        if 'correct' in judgment_lower and 'incorrect' not in judgment_lower:
            return True
        elif 'incorrect' in judgment_lower:
            return False
        else:
            # Default to False if unclear
            return False
    
    def _calculate_confidence(
        self, 
        response_time: float, 
        judgment_text: str
    ) -> float:
        """Calculate confidence based on response time and judgment clarity."""
        # Time-based confidence (faster = more confident)
        time_confidence = max(0.0, 1.0 - (response_time / self.time_limit))
        
        # Clarity-based confidence (clearer judgment = more confident)
        clarity_confidence = 1.0 if 'correct' in judgment_text.lower() or 'incorrect' in judgment_text.lower() else 0.5
        
        # Combined confidence
        return (time_confidence + clarity_confidence) / 2.0
    
    def evaluate_batch(
        self,
        problems: List[str],
        solutions: List[str],
        ground_truths: List[str],
        evaluator_id: str = "proxy"
    ) -> List[HumanEvaluationResult]:
        """
        Evaluate a batch of solutions.
        
        Args:
            problems: List of math problems
            solutions: List of generated solutions
            ground_truths: List of ground truth answers
            evaluator_id: ID of the evaluator
            
        Returns:
            List of human evaluation results
        """
        results = []
        
        for problem, solution, ground_truth in zip(problems, solutions, ground_truths):
            try:
                result = self.evaluate_solution(
                    problem, solution, ground_truth, evaluator_id
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to evaluate solution: {e}")
                # Create a default result
                results.append(HumanEvaluationResult(
                    problem=problem,
                    solution=solution,
                    ground_truth=ground_truth,
                    human_judgment=False,
                    response_time=self.time_limit,
                    confidence=0.0,
                    evaluator_id=evaluator_id,
                    timestamp=datetime.now().isoformat()
                ))
        
        return results
    
    def calculate_human_metrics(
        self,
        results: List[HumanEvaluationResult]
    ) -> Dict[str, float]:
        """
        Calculate metrics from human evaluation results.
        
        Args:
            results: List of human evaluation results
            
        Returns:
            Dictionary with human evaluation metrics
        """
        if not results:
            return {
                'accuracy': 0.0,
                'average_response_time': 0.0,
                'average_confidence': 0.0,
                'timeout_rate': 0.0
            }
        
        # Calculate accuracy
        correct_judgments = sum(1 for r in results if r.human_judgment)
        accuracy = correct_judgments / len(results)
        
        # Calculate average response time
        average_response_time = np.mean([r.response_time for r in results])
        
        # Calculate average confidence
        average_confidence = np.mean([r.confidence for r in results])
        
        # Calculate timeout rate
        timeouts = sum(1 for r in results if r.response_time >= self.time_limit)
        timeout_rate = timeouts / len(results)
        
        return {
            'accuracy': accuracy,
            'average_response_time': average_response_time,
            'average_confidence': average_confidence,
            'timeout_rate': timeout_rate
        }
    
    def evaluate_legibility_improvement(
        self,
        round_results: Dict[int, List[HumanEvaluationResult]]
    ) -> Dict[str, float]:
        """
        Evaluate how legibility improves over training rounds.
        
        Args:
            round_results: Dictionary mapping rounds to human evaluation results
            
        Returns:
            Dictionary with legibility improvement metrics
        """
        improvement_metrics = {}
        
        # Calculate improvement for each metric
        metrics = ['accuracy', 'average_confidence', 'timeout_rate']
        
        for metric in metrics:
            if len(round_results) > 1:
                initial_values = []
                final_values = []
                
                for round_num, results in round_results.items():
                    round_metrics = self.calculate_human_metrics(results)
                    if round_num == 0:
                        initial_values.append(round_metrics[metric])
                    elif round_num == len(round_results) - 1:
                        final_values.append(round_metrics[metric])
                
                if initial_values and final_values:
                    initial_avg = np.mean(initial_values)
                    final_avg = np.mean(final_values)
                    improvement = final_avg - initial_avg
                    improvement_metrics[f'{metric}_improvement'] = improvement
        
        # Calculate overall legibility improvement
        if 'accuracy_improvement' in improvement_metrics and 'timeout_rate_improvement' in improvement_metrics:
            # Positive accuracy improvement and negative timeout rate improvement
            overall_improvement = (
                improvement_metrics['accuracy_improvement'] - 
                improvement_metrics['timeout_rate_improvement']
            )
            improvement_metrics['overall_legibility_improvement'] = overall_improvement
        
        return improvement_metrics
    
    def save_evaluation_results(
        self,
        results: List[HumanEvaluationResult],
        save_path: str
    ):
        """Save human evaluation results to a file."""
        data = []
        for result in results:
            data.append({
                'problem': result.problem,
                'solution': result.solution,
                'ground_truth': result.ground_truth,
                'human_judgment': result.human_judgment,
                'response_time': result.response_time,
                'confidence': result.confidence,
                'evaluator_id': result.evaluator_id,
                'timestamp': result.timestamp
            })
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved human evaluation results to {save_path}")
    
    def load_evaluation_results(self, load_path: str) -> List[HumanEvaluationResult]:
        """Load human evaluation results from a file."""
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        results = []
        for item in data:
            result = HumanEvaluationResult(
                problem=item['problem'],
                solution=item['solution'],
                ground_truth=item['ground_truth'],
                human_judgment=item['human_judgment'],
                response_time=item['response_time'],
                confidence=item['confidence'],
                evaluator_id=item['evaluator_id'],
                timestamp=item['timestamp']
            )
            results.append(result)
        
        logger.info(f"Loaded {len(results)} human evaluation results from {load_path}")
        return results
    
    def generate_legibility_report(
        self,
        round_results: Dict[int, List[HumanEvaluationResult]],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a legibility evaluation report.
        
        Args:
            round_results: Dictionary mapping rounds to human evaluation results
            save_path: Optional path to save the report
            
        Returns:
            Report text
        """
        report = "# Human Legibility Evaluation Report\n\n"
        
        # Summary table
        report += "## Legibility Summary\n\n"
        report += "| Round | Accuracy | Avg Response Time | Avg Confidence | Timeout Rate |\n"
        report += "|-------|----------|-------------------|----------------|-------------|\n"
        
        for round_num, results in round_results.items():
            metrics = self.calculate_human_metrics(results)
            report += f"| {round_num} | {metrics['accuracy']:.3f} | {metrics['average_response_time']:.2f}s | {metrics['average_confidence']:.3f} | {metrics['timeout_rate']:.3f} |\n"
        
        # Improvement analysis
        report += "\n## Legibility Improvement\n\n"
        improvement_metrics = self.evaluate_legibility_improvement(round_results)
        
        for metric, value in improvement_metrics.items():
            report += f"- **{metric}**: {value:.3f}\n"
        
        # Detailed analysis
        report += "\n## Detailed Analysis\n\n"
        
        for round_num, results in round_results.items():
            report += f"### Round {round_num}\n\n"
            
            metrics = self.calculate_human_metrics(results)
            report += f"- **Accuracy**: {metrics['accuracy']:.3f}\n"
            report += f"- **Average Response Time**: {metrics['average_response_time']:.2f}s\n"
            report += f"- **Average Confidence**: {metrics['average_confidence']:.3f}\n"
            report += f"- **Timeout Rate**: {metrics['timeout_rate']:.3f}\n\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
