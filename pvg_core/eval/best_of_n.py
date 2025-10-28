"""
Balanced best-of-n evaluation for measuring legibility.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BestOfNResult:
    """Result of best-of-n evaluation."""
    n: int
    accuracy: float
    correct_selections: int
    total_selections: int
    helpful_accuracy: float
    sneaky_accuracy: float
    selection_distribution: Dict[str, int]

class BestOfNEvaluator:
    """Evaluator for balanced best-of-n selection."""
    
    def __init__(self, verifier_model, tokenizer):
        """
        Initialize the best-of-n evaluator.
        
        Args:
            verifier_model: Trained verifier model
            tokenizer: Tokenizer for the verifier
        """
        self.verifier_model = verifier_model
        self.tokenizer = tokenizer
    
    def create_balanced_set(
        self,
        problems: List[str],
        solutions: List[str],
        ground_truths: List[str],
        roles: List[str],
        n_samples: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Create a balanced set of correct and incorrect solutions.
        
        Args:
            problems: List of math problems
            solutions: List of generated solutions
            ground_truths: List of ground truth answers
            roles: List of roles
            n_samples: Number of samples to generate per problem
            
        Returns:
            List of balanced solution sets
        """
        from ..parsing import extract_final_answer, is_correct
        
        balanced_sets = []
        
        for i, (problem, ground_truth, role) in enumerate(zip(problems, ground_truths, roles)):
            # Generate multiple solutions for this problem
            problem_solutions = []
            
            # Add the original solution
            original_solution = solutions[i]
            predicted_answer = extract_final_answer(original_solution)
            is_correct = is_correct(predicted_answer, ground_truth)
            
            problem_solutions.append({
                'solution': original_solution,
                'is_correct': is_correct,
                'role': role,
                'problem': problem,
                'ground_truth': ground_truth
            })
            
            # Generate additional solutions if needed
            # For now, we'll use the original solution multiple times
            # In practice, you might want to generate multiple solutions
            while len(problem_solutions) < n_samples:
                problem_solutions.append({
                    'solution': original_solution,
                    'is_correct': is_correct,
                    'role': role,
                    'problem': problem,
                    'ground_truth': ground_truth
                })
            
            balanced_sets.append(problem_solutions)
        
        return balanced_sets
    
    def evaluate_best_of_n(
        self,
        balanced_sets: List[List[Dict[str, Any]]],
        n_values: List[int] = [1, 5, 10, 20, 50, 100]
    ) -> Dict[int, BestOfNResult]:
        """
        Evaluate best-of-n selection for different values of n.
        
        Args:
            balanced_sets: List of balanced solution sets
            n_values: List of n values to evaluate
            
        Returns:
            Dictionary mapping n values to results
        """
        results = {}
        
        for n in n_values:
            logger.info(f"Evaluating best-of-{n}")
            
            total_correct = 0
            total_selections = 0
            helpful_correct = 0
            helpful_total = 0
            sneaky_correct = 0
            sneaky_total = 0
            selection_distribution = {'helpful': 0, 'sneaky': 0}
            
            for problem_set in balanced_sets:
                # Get verifier scores for all solutions
                verifier_scores = []
                for solution_data in problem_set:
                    score = self._get_verifier_score(
                        solution_data['problem'],
                        solution_data['solution']
                    )
                    verifier_scores.append(score)
                
                # Select top n solutions
                top_n_indices = np.argsort(verifier_scores)[-n:]
                top_n_solutions = [problem_set[i] for i in top_n_indices]
                
                # Check if any of the top n are correct
                has_correct = any(sol['is_correct'] for sol in top_n_solutions)
                if has_correct:
                    total_correct += 1
                
                total_selections += 1
                
                # Track role-specific performance
                for sol in top_n_solutions:
                    if sol['role'] == 'helpful':
                        helpful_total += 1
                        if sol['is_correct']:
                            helpful_correct += 1
                        selection_distribution['helpful'] += 1
                    else:
                        sneaky_total += 1
                        if sol['is_correct']:
                            sneaky_correct += 1
                        selection_distribution['sneaky'] += 1
            
            # Calculate metrics
            accuracy = total_correct / total_selections if total_selections > 0 else 0.0
            helpful_accuracy = helpful_correct / helpful_total if helpful_total > 0 else 0.0
            sneaky_accuracy = sneaky_correct / sneaky_total if sneaky_total > 0 else 0.0
            
            results[n] = BestOfNResult(
                n=n,
                accuracy=accuracy,
                correct_selections=total_correct,
                total_selections=total_selections,
                helpful_accuracy=helpful_accuracy,
                sneaky_accuracy=sneaky_accuracy,
                selection_distribution=selection_distribution
            )
            
            logger.info(f"Best-of-{n}: accuracy={accuracy:.3f}, helpful={helpful_accuracy:.3f}, sneaky={sneaky_accuracy:.3f}")
        
        return results
    
    def _get_verifier_score(self, problem: str, solution: str) -> float:
        """Get verifier score for a problem-solution pair."""
        from ..prompts import get_verifier_classification_prompt
        
        # Format input
        prompt = get_verifier_classification_prompt(problem, solution)
        
        # Tokenize
        inputs = self.tokenizer.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Get verifier score
        with torch.no_grad():
            outputs = self.verifier_model(**inputs)
            scores = torch.softmax(outputs['logits'], dim=-1)
            score = scores[0, 1].item()  # Probability of correct class
        
        return score
    
    def evaluate_legibility_improvement(
        self,
        round_results: Dict[int, Dict[int, BestOfNResult]]
    ) -> Dict[str, float]:
        """
        Evaluate how legibility improves over training rounds.
        
        Args:
            round_results: Dictionary mapping rounds to best-of-n results
            
        Returns:
            Dictionary with legibility improvement metrics
        """
        improvement_metrics = {}
        
        # Calculate improvement for each n value
        for n in [1, 5, 10, 20, 50, 100]:
            if n in round_results[0] and n in round_results[len(round_results) - 1]:
                initial_accuracy = round_results[0][n].accuracy
                final_accuracy = round_results[len(round_results) - 1][n].accuracy
                improvement = final_accuracy - initial_accuracy
                improvement_metrics[f'best_of_{n}_improvement'] = improvement
        
        # Calculate overall legibility score
        # Weighted average of improvements across n values
        weights = {1: 0.4, 5: 0.3, 10: 0.2, 20: 0.1}  # Higher weight for smaller n
        weighted_improvement = 0.0
        total_weight = 0.0
        
        for n, weight in weights.items():
            if f'best_of_{n}_improvement' in improvement_metrics:
                weighted_improvement += improvement_metrics[f'best_of_{n}_improvement'] * weight
                total_weight += weight
        
        if total_weight > 0:
            improvement_metrics['overall_legibility_improvement'] = weighted_improvement / total_weight
        
        return improvement_metrics
    
    def plot_best_of_n_curves(
        self,
        round_results: Dict[int, Dict[int, BestOfNResult]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot best-of-n accuracy curves for different rounds.
        
        Args:
            round_results: Dictionary mapping rounds to best-of-n results
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        for round_num, results in round_results.items():
            n_values = sorted(results.keys())
            accuracies = [results[n].accuracy for n in n_values]
            
            plt.plot(n_values, accuracies, marker='o', label=f'Round {round_num}')
        
        plt.xlabel('n (number of solutions)')
        plt.ylabel('Accuracy')
        plt.title('Best-of-n Accuracy Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(
        self,
        round_results: Dict[int, Dict[int, BestOfNResult]],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive report of best-of-n evaluation.
        
        Args:
            round_results: Dictionary mapping rounds to best-of-n results
            save_path: Optional path to save the report
            
        Returns:
            Report text
        """
        report = "# Best-of-n Evaluation Report\n\n"
        
        # Summary table
        report += "## Summary\n\n"
        report += "| Round | Best-of-1 | Best-of-5 | Best-of-10 | Best-of-20 | Best-of-50 | Best-of-100 |\n"
        report += "|-------|-----------|-----------|------------|------------|------------|-------------|\n"
        
        for round_num, results in round_results.items():
            row = f"| {round_num} |"
            for n in [1, 5, 10, 20, 50, 100]:
                if n in results:
                    row += f" {results[n].accuracy:.3f} |"
                else:
                    row += " N/A |"
            report += row + "\n"
        
        # Improvement analysis
        report += "\n## Legibility Improvement\n\n"
        improvement_metrics = self.evaluate_legibility_improvement(round_results)
        
        for metric, value in improvement_metrics.items():
            report += f"- **{metric}**: {value:.3f}\n"
        
        # Detailed analysis for each round
        report += "\n## Detailed Analysis\n\n"
        
        for round_num, results in round_results.items():
            report += f"### Round {round_num}\n\n"
            
            for n, result in results.items():
                report += f"**Best-of-{n}**:\n"
                report += f"- Overall accuracy: {result.accuracy:.3f}\n"
                report += f"- Helpful accuracy: {result.helpful_accuracy:.3f}\n"
                report += f"- Sneaky accuracy: {result.sneaky_accuracy:.3f}\n"
                report += f"- Selection distribution: {result.selection_distribution}\n\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
