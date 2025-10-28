#!/usr/bin/env python3
"""
Train sneaky-only attacks against a frozen verifier.
"""

import argparse
import json
import os
import logging
from typing import Dict, Any, List
import torch
import numpy as np
from omegaconf import OmegaConf

# Import our modules
from pvg_core.verifier import VerifierModel, VerifierTokenizer
from pvg_core.prover import ProverPolicyLoader
from pvg_core.eval.attacks import SneakyOnlyAttacker, AttackEvaluator
from pvg_core.rewards import RewardConfig, RewardCalculator

logger = logging.getLogger(__name__)

def load_models(model_dir: str, device: str = "cuda"):
    """Load trained models."""
    logger.info(f"Loading models from {model_dir}")
    
    # Load verifier
    verifier_model = VerifierModel.from_pretrained(f"{model_dir}/verifier")
    verifier_tokenizer = VerifierTokenizer(
        tokenizer_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-503b",
        max_length=512
    )
    
    # Load prover
    prover_loader = ProverPolicyLoader(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        use_peft=True
    )
    prover_loader.load_pretrained(f"{model_dir}/prover")
    
    return verifier_model, verifier_tokenizer, prover_loader

def train_sneaky_attacks(
    prover_loader: ProverPolicyLoader,
    verifier_model: VerifierModel,
    verifier_tokenizer: VerifierTokenizer,
    test_data: List[Dict[str, Any]],
    reward_type: str = "goodhart",
    max_attempts: int = 10,
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """Train sneaky attacks against the verifier."""
    logger.info(f"Training sneaky attacks with reward type: {reward_type}")
    
    # Create reward calculator
    reward_config = RewardConfig(reward_type=reward_type)
    reward_calculator = RewardCalculator(reward_config)
    
    # Create attacker
    attacker = SneakyOnlyAttacker(
        prover_loader.get_model(),
        verifier_model,
        prover_loader.get_tokenizer(),
        device=device
    )
    
    # Prepare test data
    problems = [item['problem'] for item in test_data]
    ground_truths = [item['ground_truth'] for item in test_data]
    
    # Generate sneaky attacks
    attack_results = attacker.generate_sneaky_solutions(
        problems, ground_truths, max_attempts=max_attempts
    )
    
    # Evaluate attack success
    success_metrics = attacker.evaluate_attack_success(attack_results)
    
    logger.info(f"Attack success metrics: {success_metrics}")
    
    return attack_results

def evaluate_verifier_robustness(
    verifier_model: VerifierModel,
    verifier_tokenizer: VerifierTokenizer,
    attack_results: List[Dict[str, Any]],
    device: str = "cuda"
) -> Dict[str, float]:
    """Evaluate verifier robustness against attacks."""
    logger.info("Evaluating verifier robustness...")
    
    # Create evaluator
    evaluator = AttackEvaluator(verifier_model, verifier_tokenizer, device=device)
    
    # Evaluate robustness
    robustness_metrics = evaluator.evaluate_verifier_robustness(attack_results)
    
    logger.info(f"Robustness metrics: {robustness_metrics}")
    
    return robustness_metrics

def compare_reward_functions(
    prover_loader: ProverPolicyLoader,
    verifier_model: VerifierModel,
    verifier_tokenizer: VerifierTokenizer,
    test_data: List[Dict[str, Any]],
    reward_types: List[str] = ["src", "cgc", "goodhart"],
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """Compare different reward functions for attacks."""
    logger.info("Comparing reward functions...")
    
    results = {}
    
    for reward_type in reward_types:
        logger.info(f"Testing reward type: {reward_type}")
        
        # Create reward calculator
        reward_config = RewardConfig(reward_type=reward_type)
        reward_calculator = RewardCalculator(reward_config)
        
        # Create attacker
        attacker = SneakyOnlyAttacker(
            prover_loader.get_model(),
            verifier_model,
            prover_loader.get_tokenizer(),
            device=device
        )
        
        # Prepare test data
        problems = [item['problem'] for item in test_data[:5]]  # Use subset
        ground_truths = [item['ground_truth'] for item in test_data[:5]]
        
        # Generate attacks
        attack_results = attacker.generate_sneaky_solutions(
            problems, ground_truths, max_attempts=5
        )
        
        # Evaluate success
        success_metrics = attacker.evaluate_attack_success(attack_results)
        
        # Evaluate robustness
        robustness_metrics = evaluate_verifier_robustness(
            verifier_model, verifier_tokenizer, attack_results, device
        )
        
        results[reward_type] = {
            'success_metrics': success_metrics,
            'robustness_metrics': robustness_metrics
        }
    
    return results

def generate_attack_report(
    attack_results: List[Dict[str, Any]],
    robustness_metrics: Dict[str, float],
    comparison_results: Dict[str, Dict[str, float]],
    save_path: str
) -> str:
    """Generate a comprehensive attack report."""
    report = "# Sneaky-Only Attack Report\n\n"
    
    # Summary
    report += "## Attack Summary\n\n"
    report += f"- **Total Attacks**: {len(attack_results)}\n"
    report += f"- **Average Verifier Score**: {np.mean([r['verifier_score'] for r in attack_results]):.3f}\n"
    report += f"- **Max Verifier Score**: {max([r['verifier_score'] for r in attack_results]):.3f}\n"
    report += f"- **Min Verifier Score**: {min([r['verifier_score'] for r in attack_results]):.3f}\n\n"
    
    # Robustness metrics
    report += "## Verifier Robustness\n\n"
    report += f"- **Robustness Score**: {robustness_metrics['robustness_score']:.3f}\n"
    report += f"- **False Positive Rate**: {robustness_metrics['false_positive_rate']:.3f}\n"
    report += f"- **Average Confidence**: {robustness_metrics['average_confidence']:.3f}\n"
    report += f"- **High Confidence Attacks**: {robustness_metrics['high_confidence_attacks']}\n\n"
    
    # Reward function comparison
    report += "## Reward Function Comparison\n\n"
    report += "| Reward Type | Success Rate | Robustness Score | False Positive Rate |\n"
    report += "|-------------|--------------|------------------|---------------------|\n"
    
    for reward_type, metrics in comparison_results.items():
        success_rate = metrics['success_metrics']['success_rate']
        robustness_score = metrics['robustness_metrics']['robustness_score']
        false_positive_rate = metrics['robustness_metrics']['false_positive_rate']
        report += f"| {reward_type} | {success_rate:.3f} | {robustness_score:.3f} | {false_positive_rate:.3f} |\n"
    
    # Detailed attack results
    report += "\n## Detailed Attack Results\n\n"
    for i, attack in enumerate(attack_results[:10]):  # Show first 10
        report += f"### Attack {i+1}\n\n"
        report += f"**Problem**: {attack['problem'][:100]}...\n"
        report += f"**Solution**: {attack['solution'][:200]}...\n"
        report += f"**Verifier Score**: {attack['verifier_score']:.3f}\n"
        report += f"**Attempts**: {attack['attempt']}\n\n"
    
    # Save report
    with open(save_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Attack report saved to {save_path}")
    return report

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train sneaky-only attacks")
    parser.add_argument("--model-dir", required=True, help="Directory containing trained models")
    parser.add_argument("--data-dir", default="data/gsm8k", help="Data directory")
    parser.add_argument("--output-dir", default="outputs/attacks", help="Output directory")
    parser.add_argument("--reward", default="goodhart", choices=["src", "cgc", "goodhart"], help="Reward function to test")
    parser.add_argument("--max-attempts", type=int, default=10, help="Maximum attempts per problem")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    verifier_model, verifier_tokenizer, prover_loader = load_models(args.model_dir, args.device)
    
    # Load test data
    test_data = []
    test_path = os.path.join(args.data_dir, "test.jsonl")
    if os.path.exists(test_path):
        with open(test_path, 'r') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
    else:
        logger.warning(f"Test data not found at {test_path}")
        return
    
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Train sneaky attacks
    attack_results = train_sneaky_attacks(
        prover_loader, verifier_model, verifier_tokenizer, test_data,
        reward_type=args.reward, max_attempts=args.max_attempts, device=args.device
    )
    
    # Evaluate verifier robustness
    robustness_metrics = evaluate_verifier_robustness(
        verifier_model, verifier_tokenizer, attack_results, args.device
    )
    
    # Compare reward functions
    comparison_results = compare_reward_functions(
        prover_loader, verifier_model, verifier_tokenizer, test_data, device=args.device
    )
    
    # Save results
    results = {
        'attack_results': attack_results,
        'robustness_metrics': robustness_metrics,
        'comparison_results': comparison_results
    }
    
    results_path = os.path.join(args.output_dir, "attack_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    report_path = os.path.join(args.output_dir, "attack_report.md")
    report = generate_attack_report(
        attack_results, robustness_metrics, comparison_results, report_path
    )
    
    logger.info(f"Attack training complete! Results saved to {args.output_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("ATTACK SUMMARY")
    print("="*50)
    print(f"Total Attacks: {len(attack_results)}")
    print(f"Average Verifier Score: {np.mean([r['verifier_score'] for r in attack_results]):.3f}")
    print(f"Robustness Score: {robustness_metrics['robustness_score']:.3f}")
    print(f"False Positive Rate: {robustness_metrics['false_positive_rate']:.3f}")
    
    print("\nReward Function Comparison:")
    for reward_type, metrics in comparison_results.items():
        success_rate = metrics['success_metrics']['success_rate']
        robustness_score = metrics['robustness_metrics']['robustness_score']
        print(f"  {reward_type}: Success={success_rate:.3f}, Robustness={robustness_score:.3f}")

if __name__ == "__main__":
    main()
