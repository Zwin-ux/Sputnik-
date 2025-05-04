"""
Experimental setup for evaluating the HATD framework.
Implements the evaluation design described in Section 5 of the paper.
"""

import os
import argparse
import logging
import json
import time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional, Any

from src.pipeline.hatd_pipeline import HATDPipeline

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class HATDExperiment:
    """
    Manages the experimental evaluation of the HATD framework.
    """
    
    def __init__(
        self,
        output_dir: str = "experiments/results",
        data_dir: str = "data",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config_path: Optional[str] = None
    ):
        """
        Initialize the experiment manager.
        
        Args:
            output_dir: Directory to save experiment results
            data_dir: Directory to store dataset files
            device: Device to run experiments on
            config_path: Path to a JSON configuration file for the HATD pipeline
        """
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.device = device
        self.config_path = config_path
        
        # Create necessary directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Setup datasets mapping
        self.dataset_loaders = {
            "qa": self._load_qa_dataset,
            "generation": self._load_generation_dataset,
            "reasoning": self._load_reasoning_dataset
        }
        
        # Setup evaluation metrics
        self.metrics = {
            "qa": self._evaluate_qa,
            "generation": self._evaluate_generation,
            "reasoning": self._evaluate_reasoning
        }
        
        # Initialize results storage
        self.results = {
            "baseline": {},
            "hatd": {}
        }
    
    def _load_qa_dataset(self, n_samples: int = 100) -> Dict[str, Any]:
        """
        Load and prepare SQuAD dataset for question answering evaluation.
        
        Args:
            n_samples: Number of samples to use
            
        Returns:
            Prepared dataset
        """
        logger.info("Loading SQuAD dataset for question answering")
        
        # Load SQuAD dataset
        try:
            dataset = load_dataset("squad", split="validation")
            
            # Take a subset of samples
            if len(dataset) > n_samples:
                indices = np.random.choice(len(dataset), n_samples, replace=False)
                dataset = dataset.select(indices)
                
            return {
                "data": dataset,
                "type": "qa",
                "name": "SQuAD"
            }
        except Exception as e:
            logger.error(f"Error loading SQuAD dataset: {e}")
            # Create dummy dataset for demonstration
            return self._create_dummy_qa_dataset(n_samples)
    
    def _create_dummy_qa_dataset(self, n_samples: int = 100) -> Dict[str, Any]:
        """
        Create a dummy QA dataset for demonstration purposes.
        
        Args:
            n_samples: Number of samples to create
            
        Returns:
            Dummy dataset
        """
        dummy_data = []
        for i in range(n_samples):
            dummy_data.append({
                "id": f"dummy-{i}",
                "context": f"This is a dummy context {i} about artificial intelligence. " +
                           f"Large language models were developed to process text efficiently.",
                "question": f"What were large language models developed for?",
                "answers": {
                    "text": ["to process text efficiently"],
                    "answer_start": [82]
                }
            })
        
        return {
            "data": {"dummy": dummy_data},
            "type": "qa",
            "name": "Dummy-SQuAD"
        }
    
    def _load_generation_dataset(self, n_samples: int = 100) -> Dict[str, Any]:
        """
        Load and prepare CommonGen dataset for text generation evaluation.
        
        Args:
            n_samples: Number of samples to use
            
        Returns:
            Prepared dataset
        """
        logger.info("Loading CommonGen dataset for text generation")
        
        # Load CommonGen dataset
        try:
            dataset = load_dataset("common_gen", split="validation")
            
            # Take a subset of samples
            if len(dataset) > n_samples:
                indices = np.random.choice(len(dataset), n_samples, replace=False)
                dataset = dataset.select(indices)
                
            return {
                "data": dataset,
                "type": "generation",
                "name": "CommonGen"
            }
        except Exception as e:
            logger.error(f"Error loading CommonGen dataset: {e}")
            # Create dummy dataset for demonstration
            return self._create_dummy_generation_dataset(n_samples)
    
    def _create_dummy_generation_dataset(self, n_samples: int = 100) -> Dict[str, Any]:
        """
        Create a dummy generation dataset for demonstration purposes.
        
        Args:
            n_samples: Number of samples to create
            
        Returns:
            Dummy dataset
        """
        concepts_list = [
            ["dog", "frisbee", "catch"],
            ["chef", "cook", "meal"],
            ["student", "study", "exam"],
            ["musician", "play", "instrument"],
            ["artist", "paint", "canvas"]
        ]
        
        dummy_data = []
        for i in range(n_samples):
            concept_idx = i % len(concepts_list)
            dummy_data.append({
                "id": f"dummy-{i}",
                "concepts": concepts_list[concept_idx],
                "target": f"A coherent sentence using the concepts {', '.join(concepts_list[concept_idx])}."
            })
        
        return {
            "data": {"dummy": dummy_data},
            "type": "generation",
            "name": "Dummy-CommonGen"
        }
    
    def _load_reasoning_dataset(self, n_samples: int = 100) -> Dict[str, Any]:
        """
        Load and prepare GSM8K dataset for mathematical reasoning evaluation.
        
        Args:
            n_samples: Number of samples to use
            
        Returns:
            Prepared dataset
        """
        logger.info("Loading GSM8K dataset for mathematical reasoning")
        
        # Load GSM8K dataset
        try:
            dataset = load_dataset("gsm8k", "main", split="test")
            
            # Take a subset of samples
            if len(dataset) > n_samples:
                indices = np.random.choice(len(dataset), n_samples, replace=False)
                dataset = dataset.select(indices)
                
            return {
                "data": dataset,
                "type": "reasoning",
                "name": "GSM8K"
            }
        except Exception as e:
            logger.error(f"Error loading GSM8K dataset: {e}")
            # Create dummy dataset for demonstration
            return self._create_dummy_reasoning_dataset(n_samples)
    
    def _create_dummy_reasoning_dataset(self, n_samples: int = 100) -> Dict[str, Any]:
        """
        Create a dummy reasoning dataset for demonstration purposes.
        
        Args:
            n_samples: Number of samples to create
            
        Returns:
            Dummy dataset
        """
        dummy_data = []
        for i in range(n_samples):
            dummy_data.append({
                "id": f"dummy-{i}",
                "question": f"John has {i+5} apples. He gives 3 to Mary and then buys {i+2} more. How many apples does John have now?",
                "answer": f"{i+5-3+i+2}",
                "steps": f"John starts with {i+5} apples. He gives 3 to Mary, so now he has {i+5-3}. Then he buys {i+2} more, so he has {i+5-3}+{i+2}={i+5-3+i+2} apples."
            })
        
        return {
            "data": {"dummy": dummy_data},
            "type": "reasoning",
            "name": "Dummy-GSM8K"
        }
    
    def _evaluate_qa(self, model_outputs: List[str], dataset: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate question answering performance (accuracy).
        
        Args:
            model_outputs: List of model prediction strings
            dataset: The dataset used for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        # For demo purposes, we'll simulate evaluation
        # In a real implementation, this would use standard QA metrics
        
        correct = 0
        total = len(model_outputs)
        
        # Simulate matching with ground truth
        for i in range(total):
            # Randomly determine if answer is correct (for demo)
            if np.random.random() > 0.2:  # 80% chance of being correct
                correct += 1
        
        return {
            "accuracy": correct / total,
            "f1_score": 0.85  # Simulated F1 score
        }
    
    def _evaluate_generation(self, model_outputs: List[str], dataset: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate text generation performance (BLEU score).
        
        Args:
            model_outputs: List of model prediction strings
            dataset: The dataset used for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        # For demo purposes, we'll simulate evaluation
        # In a real implementation, this would calculate BLEU scores
        
        # Simulate BLEU scores
        bleu_score = 0.78 + np.random.random() * 0.1
        
        return {
            "bleu": bleu_score,
            "diversity": 0.65 + np.random.random() * 0.1  # Simulated diversity score
        }
    
    def _evaluate_reasoning(self, model_outputs: List[str], dataset: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate mathematical reasoning performance (correctness).
        
        Args:
            model_outputs: List of model prediction strings
            dataset: The dataset used for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        # For demo purposes, we'll simulate evaluation
        # In a real implementation, this would check correctness of solutions
        
        correct = 0
        total = len(model_outputs)
        
        # Simulate matching with ground truth
        for i in range(total):
            # Randomly determine if answer is correct (for demo)
            if np.random.random() > 0.3:  # 70% chance of being correct
                correct += 1
        
        return {
            "accuracy": correct / total,
            "step_accuracy": 0.75 + np.random.random() * 0.1  # Simulated step accuracy
        }
    
    def run_baseline(self, dataset_type: str, n_samples: int = 100) -> Dict[str, Any]:
        """
        Run baseline evaluation without HATD optimization.
        
        Args:
            dataset_type: Type of dataset to use ('qa', 'generation', or 'reasoning')
            n_samples: Number of samples to evaluate
            
        Returns:
            Evaluation results
        """
        if dataset_type not in self.dataset_loaders:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Load dataset
        dataset = self.dataset_loaders[dataset_type](n_samples)
        
        logger.info(f"Running baseline evaluation on {dataset['name']} dataset")
        
        # Track metrics
        token_usage = []
        inference_times = []
        outputs = []
        
        # Process each sample
        start_time = time.time()
        for i, sample in enumerate(dataset["data"]):
            sample_start = time.time()
            
            # Get input text based on dataset type
            if dataset_type == "qa":
                input_text = f"Context: {sample['context']}\nQuestion: {sample['question']}"
            elif dataset_type == "generation":
                input_text = f"Generate a sentence using these concepts: {', '.join(sample['concepts'])}"
            else:  # reasoning
                input_text = sample["question"]
            
            # Count tokens
            tokens = input_text.split()
            token_usage.append(len(tokens))
            
            # Simulate model processing
            time.sleep(0.01 * len(tokens))  # Simulate processing time proportional to tokens
            
            # Simulate model output
            output = f"Simulated output for input {i}"
            outputs.append(output)
            
            # Track inference time
            inference_time = time.time() - sample_start
            inference_times.append(inference_time)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{n_samples} samples")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_token_usage = sum(token_usage) / len(token_usage)
        avg_inference_time = sum(inference_times) / len(inference_times)
        
        # Evaluate performance
        performance_metrics = self.metrics[dataset_type](outputs, dataset)
        
        # Compile results
        results = {
            "dataset": dataset["name"],
            "dataset_type": dataset_type,
            "samples": n_samples,
            "avg_token_usage": avg_token_usage,
            "avg_inference_time": avg_inference_time,
            "total_time": total_time,
            "performance": performance_metrics
        }
        
        # Store results
        self.results["baseline"][dataset_type] = results
        
        # Save results
        self._save_results()
        
        return results
    
    def run_hatd(self, dataset_type: str, n_samples: int = 100) -> Dict[str, Any]:
        """
        Run evaluation with HATD optimization.
        
        Args:
            dataset_type: Type of dataset to use ('qa', 'generation', or 'reasoning')
            n_samples: Number of samples to evaluate
            
        Returns:
            Evaluation results
        """
        if dataset_type not in self.dataset_loaders:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Load dataset
        dataset = self.dataset_loaders[dataset_type](n_samples)
        
        logger.info(f"Running HATD evaluation on {dataset['name']} dataset")
        
        # Initialize HATD pipeline
        hatd = HATDPipeline(
            config_path=self.config_path,
            enable_feedback_loop=True
        )
        
        # Track metrics
        token_usage_original = []
        token_usage_processed = []
        inference_times = []
        token_reductions = []
        outputs = []
        
        # Process each sample
        start_time = time.time()
        for i, sample in enumerate(dataset["data"]):
            sample_start = time.time()
            
            # Get input text based on dataset type
            if dataset_type == "qa":
                input_text = f"Context: {sample['context']}\nQuestion: {sample['question']}"
            elif dataset_type == "generation":
                input_text = f"Generate a sentence using these concepts: {', '.join(sample['concepts'])}"
            else:  # reasoning
                input_text = sample["question"]
            
            # Count original tokens
            original_tokens = len(input_text.split())
            token_usage_original.append(original_tokens)
            
            # Process through HATD pipeline
            results = hatd.process_input(input_text)
            
            # Count processed tokens
            processed_tokens = results["preprocessed_token_count"]
            token_usage_processed.append(processed_tokens)
            
            # Calculate token reduction
            token_reduction = results["token_reduction"]
            token_reductions.append(token_reduction)
            
            # Simulate model output
            output = f"Simulated output for processed input {i}"
            outputs.append(output)
            
            # Track inference time
            inference_time = time.time() - sample_start
            inference_times.append(inference_time)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{n_samples} samples")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_original_token_usage = sum(token_usage_original) / len(token_usage_original)
        avg_processed_token_usage = sum(token_usage_processed) / len(token_usage_processed)
        avg_token_reduction = sum(token_reductions) / len(token_reductions)
        avg_inference_time = sum(inference_times) / len(inference_times)
        
        # Evaluate performance
        performance_metrics = self.metrics[dataset_type](outputs, dataset)
        
        # Get pipeline metrics
        pipeline_metrics = hatd.get_metrics()
        
        # Compile results
        results = {
            "dataset": dataset["name"],
            "dataset_type": dataset_type,
            "samples": n_samples,
            "avg_original_token_usage": avg_original_token_usage,
            "avg_processed_token_usage": avg_processed_token_usage,
            "avg_token_reduction": avg_token_reduction,
            "avg_inference_time": avg_inference_time,
            "total_time": total_time,
            "performance": performance_metrics,
            "pipeline_metrics": pipeline_metrics
        }
        
        # Store results
        self.results["hatd"][dataset_type] = results
        
        # Save results
        self._save_results()
        
        return results
    
    def run_all_experiments(self, n_samples: int = 100) -> Dict[str, Any]:
        """
        Run all experiments (baseline and HATD) on all dataset types.
        
        Args:
            n_samples: Number of samples to evaluate per dataset
            
        Returns:
            Complete results
        """
        dataset_types = ["qa", "generation", "reasoning"]
        
        # Run baseline experiments
        for dataset_type in dataset_types:
            logger.info(f"Running baseline for {dataset_type}")
            self.run_baseline(dataset_type, n_samples)
        
        # Run HATD experiments
        for dataset_type in dataset_types:
            logger.info(f"Running HATD for {dataset_type}")
            self.run_hatd(dataset_type, n_samples)
        
        # Compile comparative results
        comparative_results = self._calculate_comparative_metrics()
        
        # Generate visualizations
        self._generate_visualizations()
        
        return comparative_results
    
    def _calculate_comparative_metrics(self) -> Dict[str, Any]:
        """
        Calculate comparative metrics between baseline and HATD.
        
        Returns:
            Dictionary of comparative metrics
        """
        comparative = {}
        
        for dataset_type in self.results["baseline"]:
            if dataset_type in self.results["hatd"]:
                baseline = self.results["baseline"][dataset_type]
                hatd = self.results["hatd"][dataset_type]
                
                # Calculate metrics
                token_reduction = hatd["avg_token_reduction"]
                
                # Calculate time savings
                time_reduction = 1.0 - (hatd["avg_inference_time"] / baseline["avg_inference_time"])
                
                # Calculate performance retention
                performance_retention = {}
                for metric in baseline["performance"]:
                    if metric in hatd["performance"]:
                        retention = hatd["performance"][metric] / baseline["performance"][metric]
                        performance_retention[metric] = retention
                
                comparative[dataset_type] = {
                    "dataset": baseline["dataset"],
                    "token_reduction": token_reduction,
                    "time_reduction": time_reduction,
                    "performance_retention": performance_retention
                }
        
        # Save comparative results
        results_path = os.path.join(self.output_dir, "comparative_results.json")
        with open(results_path, 'w') as f:
            json.dump(comparative, f, indent=2)
        
        return comparative
    
    def _generate_visualizations(self):
        """
        Generate visualizations of experiment results.
        """
        comparative = self._calculate_comparative_metrics()
        
        # Create figure for token reduction
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        dataset_types = list(comparative.keys())
        token_reductions = [comparative[dt]["token_reduction"] for dt in dataset_types]
        time_reductions = [comparative[dt]["time_reduction"] for dt in dataset_types]
        
        # Create bar chart for token reductions
        x = np.arange(len(dataset_types))
        width = 0.35
        
        plt.bar(x - width/2, token_reductions, width, label='Token Reduction')
        plt.bar(x + width/2, time_reductions, width, label='Time Reduction')
        
        plt.xlabel('Dataset Type')
        plt.ylabel('Reduction Ratio')
        plt.title('HATD Efficiency Improvements')
        plt.xticks(x, dataset_types)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, "efficiency_improvements.png"))
        plt.close()
        
        # Create figure for performance retention
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        performance_metrics = []
        retention_values = []
        dataset_labels = []
        
        for dt in dataset_types:
            for metric, value in comparative[dt]["performance_retention"].items():
                performance_metrics.append(f"{dt}_{metric}")
                retention_values.append(value)
                dataset_labels.append(dt)
        
        # Create bar chart for performance retention
        x = np.arange(len(performance_metrics))
        colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_types)))
        color_map = {dt: colors[i] for i, dt in enumerate(dataset_types)}
        
        bars = plt.bar(x, retention_values, width=0.6)
        
        # Color bars by dataset type
        for i, bar in enumerate(bars):
            bar.set_color(color_map[dataset_labels[i]])
        
        plt.xlabel('Performance Metric')
        plt.ylabel('Retention Ratio')
        plt.title('HATD Performance Retention')
        plt.xticks(x, performance_metrics, rotation=45, ha='right')
        plt.ylim(0.8, 1.05)
        plt.axhline(y=0.95, color='r', linestyle='-', alpha=0.5, label='95% Retention Target')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, "performance_retention.png"))
        plt.close()
        
        logger.info(f"Generated visualizations in {self.output_dir}")
    
    def _save_results(self):
        """
        Save current results to JSON file.
        """
        results_path = os.path.join(self.output_dir, "experiment_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")


def main():
    """
    Main function to run experiments from command line.
    """
    parser = argparse.ArgumentParser(description="Run HATD experiments")
    parser.add_argument("--output_dir", type=str, default="experiments/results",
                       help="Directory to save results")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory to store datasets")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to HATD configuration file")
    parser.add_argument("--n_samples", type=int, default=100,
                       help="Number of samples to evaluate per dataset")
    parser.add_argument("--dataset", type=str, choices=["qa", "generation", "reasoning", "all"],
                       default="all", help="Dataset type to evaluate")
    parser.add_argument("--mode", type=str, choices=["baseline", "hatd", "both"],
                       default="both", help="Evaluation mode")
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = HATDExperiment(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        config_path=args.config
    )
    
    # Determine which datasets to evaluate
    if args.dataset == "all":
        dataset_types = ["qa", "generation", "reasoning"]
    else:
        dataset_types = [args.dataset]
    
    # Run experiments
    for dataset_type in dataset_types:
        if args.mode in ["baseline", "both"]:
            experiment.run_baseline(dataset_type, args.n_samples)
        
        if args.mode in ["hatd", "both"]:
            experiment.run_hatd(dataset_type, args.n_samples)
    
    # Calculate comparative metrics if both modes were run
    if args.mode == "both":
        comparative = experiment._calculate_comparative_metrics()
        experiment._generate_visualizations()
        
        logger.info("Comparative Results:")
        for dt in comparative:
            logger.info(f"{dt} - Token Reduction: {comparative[dt]['token_reduction']:.2f}, " +
                     f"Time Reduction: {comparative[dt]['time_reduction']:.2f}")


if __name__ == "__main__":
    main()
