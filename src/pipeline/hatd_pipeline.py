"""
HATD Pipeline module that integrates all components of the framework.
"""

import torch
import logging
from typing import List, Dict, Tuple, Union, Optional, Any
import os
import time
import json

from src.compression.token_scorer import TokenScorer, TokenCompressor
from src.distillation.hierarchical_distillation import HierarchicalDistiller
from src.pruning.context_sensitive_pruner import ContextSensitivePruner, QueryClassifier, RuleBased_Pruner

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class HATDPipeline:
    """
    Main pipeline for Hierarchical Adaptive Token Distillation.
    Integrates token-aware compression, hierarchical distillation, and context-sensitive pruning.
    """
    
    def __init__(
        self,
        model_hierarchy: Optional[List[str]] = None,
        compression_model: str = "bert-base-uncased",
        task_type: str = "text-classification",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "models",
        enable_feedback_loop: bool = True,
        config_path: Optional[str] = None
    ):
        """
        Initialize the HATD pipeline with all components.
        
        Args:
            model_hierarchy: List of model names in descending size order
            compression_model: Model to use for token compression
            task_type: Type of task for the distillation process
            device: Device to run the pipeline on
            output_dir: Directory to save models and artifacts
            enable_feedback_loop: Whether to enable the performance feedback loop
            config_path: Path to a JSON configuration file (overrides other arguments if provided)
        """
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Override arguments with config values
            model_hierarchy = config.get('model_hierarchy', model_hierarchy)
            compression_model = config.get('compression_model', compression_model)
            task_type = config.get('task_type', task_type)
            output_dir = config.get('output_dir', output_dir)
            enable_feedback_loop = config.get('enable_feedback_loop', enable_feedback_loop)
        
        self.device = device
        self.output_dir = output_dir
        self.enable_feedback_loop = enable_feedback_loop
        
        # Create necessary directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model hierarchy
        if model_hierarchy is None:
            model_hierarchy = HierarchicalDistiller.create_model_hierarchy(task=task_type)
        self.model_hierarchy = model_hierarchy
        
        # Initialize components
        self._init_components(compression_model, task_type)
        
        # Metrics for feedback loop
        self.metrics = {
            "token_reduction": [],
            "performance_retention": [],
            "inference_time": []
        }
        
        logger.info(f"Initialized HATD pipeline with {len(model_hierarchy)} model levels")
        logger.info(f"Model hierarchy: {model_hierarchy}")
        
    def _init_components(self, compression_model: str, task_type: str):
        """
        Initialize all pipeline components.
        
        Args:
            compression_model: Model to use for token compression
            task_type: Type of task for the distillation process
        """
        # Initialize token scoring and compression
        self.token_scorer = TokenScorer(model_name=compression_model)
        self.token_compressor = TokenCompressor(scorer=self.token_scorer)
        
        # Initialize query classification for context-sensitive pruning
        self.query_classifier = QueryClassifier()
        
        # Initialize pruning
        self.context_pruner = ContextSensitivePruner(classifier=self.query_classifier)
        self.rule_pruner = RuleBased_Pruner()
        
        # Initialize distillation components
        self.distiller = HierarchicalDistiller(
            model_hierarchy=self.model_hierarchy,
            task_type=task_type,
            output_dir=os.path.join(self.output_dir, "distilled_models")
        )
        
    def preprocess(self, text: str) -> Tuple[str, float, float]:
        """
        Preprocess input text by applying context-sensitive pruning.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Tuple of (preprocessed_text, pruning_ratio, preprocessing_time)
        """
        start_time = time.time()
        
        # Analyze context and determine pruning strategy
        features = self.context_pruner.detect_query_features(text)
        pruning_ratio = self.context_pruner.determine_pruning_ratio(text, features)
        
        # Apply rule-based pruning
        pruned_text = self.context_pruner.prune_text(
            text, 
            pruning_function=self.rule_pruner.prune,
            pruning_ratio=pruning_ratio
        )
        
        # Apply token-aware compression if needed
        if len(pruned_text.split()) > 50:  # Only apply to longer texts
            self.token_compressor.compression_ratio = pruning_ratio
            pruned_text = self.token_compressor.compress(pruned_text)
        
        preprocessing_time = time.time() - start_time
        
        return pruned_text, pruning_ratio, preprocessing_time
    
    def route_to_model(self, text: str) -> str:
        """
        Route the input to the appropriate model in the hierarchy.
        
        Args:
            text: Input text to classify
            
        Returns:
            Name of the most appropriate model for this input
        """
        # Classify query complexity
        classification = self.query_classifier.classify(text)
        complexity_label = classification["label"]
        confidence = classification["confidence"]
        
        # Determine which model to use based on complexity
        if complexity_label == "simple":
            # Use smallest model for simple queries
            model_name = self.model_hierarchy[-1]
        elif complexity_label == "medium":
            # Use middle model for medium complexity
            middle_idx = len(self.model_hierarchy) // 2
            model_name = self.model_hierarchy[middle_idx]
        else:  # complex
            # Use largest model for complex queries
            model_name = self.model_hierarchy[0]
        
        logger.info(f"Routing query with complexity '{complexity_label}' (confidence: {confidence:.2f}) to model: {model_name}")
        
        return model_name
    
    def process_input(self, text: str, track_metrics: bool = True) -> Dict[str, Any]:
        """
        Process input text through the full HATD pipeline.
        
        Args:
            text: Input text to process
            track_metrics: Whether to track performance metrics
            
        Returns:
            Dictionary with results and metrics
        """
        start_time = time.time()
        original_token_count = len(text.split())
        preprocessed_text, pruning_ratio, preprocessing_time = self.preprocess(text)
        preprocessed_token_count = len(preprocessed_text.split())
        selected_model = self.route_to_model(preprocessed_text)
        token_reduction = 1.0 - (preprocessed_token_count / original_token_count)
        
        # TODO: In a full implementation, we would process the query with the selected model
        # Here we simulate the processing time based on token count
        processing_time = 0.001 * preprocessed_token_count
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        results = {
            "original_text": text,
            "preprocessed_text": preprocessed_text,
            "original_token_count": original_token_count,
            "preprocessed_token_count": preprocessed_token_count,
            "token_reduction": token_reduction,
            "pruning_ratio": pruning_ratio,
            "selected_model": selected_model,
            "preprocessing_time": preprocessing_time,
            "processing_time": processing_time,
            "total_time": total_time
        }
        
        if track_metrics and self.enable_feedback_loop:
            self.metrics["token_reduction"].append(token_reduction)
            self.metrics["inference_time"].append(total_time)
            
            estimated_performance = max(0.95, 1.0 - (token_reduction * 0.1))
            estimated_performance = max(0.95, 1.0 - (token_reduction * 0.1))
            self.metrics["performance_retention"].append(estimated_performance)
            
            # Update feedback loop
            self._update_feedback_loop()
        
        return results
    
    def _update_feedback_loop(self):
        """
        Update pruning strategies based on accumulated metrics.
        """
        if not self.enable_feedback_loop or len(self.metrics["token_reduction"]) < 10:
            return
        
        # Calculate average metrics
        avg_token_reduction = sum(self.metrics["token_reduction"]) / len(self.metrics["token_reduction"])
        avg_performance = sum(self.metrics["performance_retention"]) / len(self.metrics["performance_retention"])
        
        # Adjust pruning strategies if performance is too low
        if avg_performance < 0.95 and avg_token_reduction > 0.2:
            # Reduce pruning aggressiveness
            for key in self.context_pruner.pruning_strategies:
                self.context_pruner.pruning_strategies[key] *= 0.9
            
            logger.info(f"Feedback loop: Reduced pruning aggressiveness due to low performance ({avg_performance:.2f})")
        
        # Adjust pruning strategies if performance is high and token reduction is low
        elif avg_performance > 0.98 and avg_token_reduction < 0.3:
            # Increase pruning aggressiveness
            for key in self.context_pruner.pruning_strategies:
                self.context_pruner.pruning_strategies[key] *= 1.1
                # Cap at reasonable maximum
                self.context_pruner.pruning_strategies[key] = min(0.7, self.context_pruner.pruning_strategies[key])
            
            logger.info(f"Feedback loop: Increased pruning aggressiveness due to high performance ({avg_performance:.2f})")
        
        # Reset metrics after adjustment
        self.metrics = {key: [] for key in self.metrics}
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metrics from the pipeline.
        
        Returns:
            Dictionary of metrics with average values
        """
        avg_metrics = {}
        
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
            else:
                avg_metrics[key] = 0.0
                
        return avg_metrics
    
    def train_distillation(self, dataset: Any, **training_kwargs) -> List[str]:
        """
        Train the hierarchical distillation pipeline.
        
        Args:
            dataset: HuggingFace dataset for training
            **training_kwargs: Additional arguments for distillation training
            
        Returns:
            List of paths to saved distilled models
        """
        return self.distiller.distill(dataset, **training_kwargs)
    
    def save_config(self, path: Optional[str] = None) -> str:
        """
        Save the pipeline configuration to a JSON file.
        
        Args:
            path: Path to save the configuration (defaults to output_dir/config.json)
            
        Returns:
            Path to the saved configuration file
        """
        if path is None:
            path = os.path.join(self.output_dir, "config.json")
            
        config = {
            "model_hierarchy": self.model_hierarchy,
            "compression_model": self.token_scorer.model_name,
            "task_type": "text-classification",  # This should be stored as an attribute in a full implementation
            "output_dir": self.output_dir,
            "enable_feedback_loop": self.enable_feedback_loop,
            "pruning_strategies": self.context_pruner.pruning_strategies,
            "device": self.device
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
            
        return path
    
    @classmethod
    def from_config(cls, config_path: str) -> 'HATDPipeline':
        """
        Create a pipeline instance from a configuration file.
        
        Args:
            config_path: Path to a JSON configuration file
            
        Returns:
            HATDPipeline instance
        """
        return cls(config_path=config_path)
