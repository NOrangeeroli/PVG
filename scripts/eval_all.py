#!/usr/bin/env python3
"""
Comprehensive evaluation of trained models.
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
from pvg_core.eval import BestOfNEvaluator, AttackEvaluator, HumanProxyEvaluator
from pvg_core.eval.metrics import calculate_comprehensive_metrics
from pvg_core.parsing import extract_final_answer, is_correct
from pvg_core.sampling import ModelSampler

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

def evaluate_verifier(
    verifier_model: VerifierModel,
    verifier_tokenizer: VerifierTokenizer,
    test_data: List[Dict[str, Any]],
    device: str = "cuda"
) -> Dict[str, float]:
    """Evaluate verifier performance."""
    logger.info("Evaluating verifier...")
    
    # Prepare test data
    problems = [item['problem'] for item in test_data]
    solutions = [item['solution'] for item in test_data]
    ground_truths = [item['ground_truth'] for item in test_data]
    
    # Get verifier scores
    verifier_scores = []
    for problem, solution in zip(problems, solutions):
        score = verifier_model.get_verifier_scores(
            [(problem, solution)]
        )[0]
        verifier_scores.append(score)
    
    # Calculate correctness
    predictions = [is_correct(extract_final_answer(sol), gt) for sol, gt in zip(solutions, ground_truths)]
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(
        predictions=predictions,
        ground_truth=ground_truths,
        scores=verifier_scores
    )
    
    logger.info(f"Verifier metrics: {metrics['basic_metrics']}")
    return metrics

def evaluate_prover(
    prover_loader: ProverPolicyLoader,
    test_data: List[Dict[str, Any]],
    device: str = "cuda"
) -> Dict[str, float]:
    """Evaluate prover performance."""
    logger.info("Evaluating prover...")
    
    # Generate solutions
    problems = [item['problem'] for item in test_data]
    ground_truths = [item['ground_truth'] for item in test_data]
    
    # Generate solutions for both roles
    helpful_problems = problems
    sneaky_problems = problems
    helpful_roles = ['helpful'] * len(problems)
    sneaky_roles = ['sneaky'] * len(problems)
    
    # Generate helpful solutions
    helpful_responses = prover_loader.generate_with_roles(
        helpful_problems, helpful_roles, max_new_tokens=512
    )
    
    # Generate sneaky solutions
    sneaky_responses = prover_loader.generate_with_roles(
        sneaky_problems, sneaky_roles, max_new_tokens=512
    )
    
    # Calculate metrics
    helpful_solutions = [r['solution'] for r in helpful_responses]
    sneaky_solutions = [r['solution'] for r in sneaky_responses]
    
    helpful_predictions = [is_correct(extract_final_answer(sol), gt) for sol, gt in zip(helpful_solutions, ground_truths)]
    sneaky_predictions = [is_correct(extract_final_answer(sol), gt) for sol, gt in zip(sneaky_solutions, ground_truths)]
    
    helpful_metrics = calculate_comprehensive_metrics(
        predictions=helpful_predictions,
        ground_truth=ground_truths,
        solutions=helpful_solutions,
        roles=['helpful'] * len(problems)
    )
    
    sneaky_metrics = calculate_comprehensive_metrics(
        predictions=sneaky_predictions,
        ground_truth=ground_truths,
        solutions=sneaky_solutions,
        roles=['sneaky'] * len(problems)
    )
    
    return {
        'helpful': helpful_metrics,
        'sneaky': sneaky_metrics
    }

def evaluate_best_of_n(
    verifier_model: VerifierModel,
    verifier_tokenizer: VerifierTokenizer,
    prover_loader: ProverPolicyLoader,
    test_data: List[Dict[str, Any]],
    n_values: List[int] = [1, 5, 10, 20, 50, 100],
    device: str = "cuda"
) -> Dict[int, Dict[str, float]]:
    """Evaluate best-of-n selection."""
    logger.info("Evaluating best-of-n selection...")
    
    # Create evaluator
    evaluator = BestOfNEvaluator(verifier_model, verifier_tokenizer)
    
    # Prepare test data
    problems = [item['problem'] for item in test_data]
    ground_truths = [item['ground_truth'] for item in test_data]
    
    # Generate solutions
    helpful_responses = prover_loader.generate_with_roles(
        problems, ['helpful'] * len(problems), max_new_tokens=512
    )
    sneaky_responses = prover_loader.generate_with_roles(
        problems, ['sneaky'] * len(problems), max_new_tokens=512
    )
    
    # Create balanced sets
    balanced_sets = evaluator.create_balanced_set(
        problems, 
        [r['solution'] for r in helpful_responses],
        ground_truths,
        ['helpful'] * len(problems),
        n_samples=20
    )
    
    # Evaluate best-of-n
    results = evaluator.evaluate_best_of_n(balanced_sets, n_values)
    
    return results

def evaluate_attacks(
    verifier_model: VerifierModel,
    verifier_tokenizer: VerifierTokenizer,
    prover_loader: ProverPolicyLoader,
    test_data: List[Dict[str, Any]],
    device: str = "cuda"
) -> Dict[str, float]:
    """Evaluate adversarial attacks."""
    logger.info("Evaluating adversarial attacks...")
    
    # Create attacker
    from pvg_core.eval.attacks import SneakyOnlyAttacker
    attacker = SneakyOnlyAttacker(
        prover_loader.get_model(),
        verifier_model,
        prover_loader.get_tokenizer(),
        device=device
    )
    
    # Prepare test data
    problems = [item['problem'] for item in test_data[:10]]  # Use subset for attacks
    ground_truths = [item['ground_truth'] for item in test_data[:10]]
    
    # Generate sneaky attacks
    attack_results = attacker.generate_sneaky_solutions(
        problems, ground_truths, max_attempts=5
    )
    
    # Evaluate attack success
    success_metrics = attacker.evaluate_attack_success(attack_results)
    
    return success_metrics

def evaluate_human_legibility(
    verifier_model: VerifierModel,
    verifier_tokenizer: VerifierTokenizer,
    prover_loader: ProverPolicyLoader,
    test_data: List[Dict[str, Any]],
    device: str = "cuda"
) -> Dict[str, float]:
    """Evaluate human legibility using proxy."""
    logger.info("Evaluating human legibility...")
    
    # Create human proxy evaluator
    evaluator = HumanProxyEvaluator(
        verifier_model,  # Use verifier as human proxy
        verifier_tokenizer,
        time_limit=45.0,
        device=device
    )
    
    # Prepare test data
    problems = [item['problem'] for item in test_data[:20]]  # Use subset
    ground_truths = [item['ground_truth'] for item in test_data[:20]]
    
    # Generate solutions
    helpful_responses = prover_loader.generate_with_roles(
        problems, ['helpful'] * len(problems), max_new_tokens=512
    )
    sneaky_responses = prover_loader.generate_with_roles(
        problems, ['sneaky'] * len(problems), max_new_tokens=512
    )
    
    # Evaluate helpful solutions
    helpful_results = evaluator.evaluate_batch(
        problems,
        [r['solution'] for r in helpful_responses],
        ground_truths,
        evaluator_id="helpful"
    )
    
    # Evaluate sneaky solutions
    sneaky_results = evaluator.evaluate_batch(
        problems,
        [r['solution'] for r in sneaky_responses],
        ground_truths,
        evaluator_id="sneaky"
    )
    
    # Calculate metrics
    helpful_metrics = evaluator.calculate_human_metrics(helpful_results)
    sneaky_metrics = evaluator.calculate_human_metrics(sneaky_results)
    
    return {
        'helpful': helpful_metrics,
        'sneaky': sneaky_metrics
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model-dir", required=True, help="Directory containing trained models")
    parser.add_argument("--data-dir", default="data/gsm8k", help="Data directory")
    parser.add_argument("--output-dir", default="outputs/eval", help="Output directory")
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
    
    # Run evaluations
    results = {}
    
    # Verifier evaluation
    try:
        verifier_results = evaluate_verifier(verifier_model, verifier_tokenizer, test_data, args.device)
        results['verifier'] = verifier_results
        logger.info("Verifier evaluation complete")
    except Exception as e:
        logger.error(f"Verifier evaluation failed: {e}")
    
    # Prover evaluation
    try:
        prover_results = evaluate_prover(prover_loader, test_data, args.device)
        results['prover'] = prover_results
        logger.info("Prover evaluation complete")
    except Exception as e:
        logger.error(f"Prover evaluation failed: {e}")
    
    # Best-of-n evaluation
    try:
        best_of_n_results = evaluate_best_of_n(
            verifier_model, verifier_tokenizer, prover_loader, test_data, args.device
        )
        results['best_of_n'] = best_of_n_results
        logger.info("Best-of-n evaluation complete")
    except Exception as e:
        logger.error(f"Best-of-n evaluation failed: {e}")
    
    # Attack evaluation
    try:
        attack_results = evaluate_attacks(
            verifier_model, verifier_tokenizer, prover_loader, test_data, args.device
        )
        results['attacks'] = attack_results
        logger.info("Attack evaluation complete")
    except Exception as e:
        logger.error(f"Attack evaluation failed: {e}")
    
    # Human legibility evaluation
    try:
        human_results = evaluate_human_legibility(
            verifier_model, verifier_tokenizer, prover_loader, test_data, args.device
        )
        results['human_legibility'] = human_results
        logger.info("Human legibility evaluation complete")
    except Exception as e:
        logger.error(f"Human legibility evaluation failed: {e}")
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation complete! Results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    if 'verifier' in results:
        print(f"Verifier Accuracy: {results['verifier']['basic_metrics']['accuracy']:.3f}")
        print(f"Verifier AUC: {results['verifier']['basic_metrics']['auc']:.3f}")
    
    if 'prover' in results:
        print(f"Helpful Accuracy: {results['prover']['helpful']['basic_metrics']['accuracy']:.3f}")
        print(f"Sneaky Accuracy: {results['prover']['sneaky']['basic_metrics']['accuracy']:.3f}")
    
    if 'best_of_n' in results:
        print(f"Best-of-1 Accuracy: {results['best_of_n'][1]['accuracy']:.3f}")
        print(f"Best-of-10 Accuracy: {results['best_of_n'][10]['accuracy']:.3f}")
    
    if 'attacks' in results:
        print(f"Attack Success Rate: {results['attacks']['success_rate']:.3f}")
    
    if 'human_legibility' in results:
        print(f"Human Accuracy (Helpful): {results['human_legibility']['helpful']['accuracy']:.3f}")
        print(f"Human Accuracy (Sneaky): {results['human_legibility']['sneaky']['accuracy']:.3f}")

if __name__ == "__main__":
    main()
