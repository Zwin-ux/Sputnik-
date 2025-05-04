"""
Hierarchical Distillation module for HATD framework.
Transfers knowledge from large teacher models to progressively smaller student models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from typing import List, Dict, Tuple, Union, Optional, Any
import os
import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class DistillationTrainer(Trainer):
    """
    Custom trainer for distillation that combines teacher and student losses.
    """
    
    def __init__(
        self, 
        teacher_model: nn.Module, 
        alpha: float = 0.5, 
        temperature: float = 2.0,
        **kwargs
    ):
        """
        Initialize the distillation trainer.
        
        Args:
            teacher_model: The teacher model to distill from
            alpha: Weight of distillation loss vs. task loss (0-1)
            temperature: Temperature for softening probability distributions
            **kwargs: Additional arguments for the Trainer base class
        """
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        
        # Ensure teacher model is in eval mode
        self.teacher_model.eval()
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute combined distillation and task loss.
        
        Args:
            model: The student model
            inputs: Input batch
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss value, or tuple of (loss, outputs) if return_outputs is True
        """
        # Get standard task loss and student outputs
        task_loss, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        # KL Divergence loss for distillation
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # Apply temperature scaling
        student_logits_t = student_logits / self.temperature
        teacher_logits_t = teacher_logits / self.temperature
        
        # Calculate distillation loss (KL divergence)
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits_t, dim=-1),
            F.softmax(teacher_logits_t, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)
        
        # Combine the losses
        loss = (1 - self.alpha) * task_loss + self.alpha * distillation_loss
        
        return (loss, student_outputs) if return_outputs else loss


class HierarchicalDistiller:
    """
    Manages the hierarchical distillation pipeline with multiple levels.
    """
    
    def __init__(
        self,
        model_hierarchy: List[str],
        task_type: str = "text-classification",
        num_labels: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "models",
    ):
        """
        Initialize the hierarchical distiller.
        
        Args:
            model_hierarchy: List of model names in descending size order (large to small)
            task_type: Type of task ('text-classification', 'question-answering', etc.)
            num_labels: Number of output labels for classification tasks
            device: Device to run training on
            output_dir: Directory to save distilled models
        """
        self.model_hierarchy = model_hierarchy
        self.task_type = task_type
        self.num_labels = num_labels
        self.device = device
        self.output_dir = output_dir
        
        # Validate model hierarchy
        if len(model_hierarchy) < 2:
            raise ValueError("Model hierarchy must contain at least 2 models (teacher and student)")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def distill(
        self,
        dataset: Any,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        alpha: float = 0.5,
        temperature: float = 2.0,
        **training_kwargs
    ):
        """
        Run the hierarchical distillation process through all levels.
        
        Args:
            dataset: HuggingFace dataset for training
            batch_size: Training batch size
            learning_rate: Learning rate for student training
            num_epochs: Number of epochs per distillation level
            alpha: Weight of distillation loss vs. task loss (0-1)
            temperature: Temperature for softening probability distributions
            **training_kwargs: Additional arguments for TrainingArguments
            
        Returns:
            List of paths to saved distilled models
        """
        saved_models = []
        
        # Iterate through teacher-student pairs in the hierarchy
        for i in range(len(self.model_hierarchy) - 1):
            teacher_name = self.model_hierarchy[i]
            student_name = self.model_hierarchy[i + 1]
            
            logger.info(f"Distilling from {teacher_name} to {student_name}")
            
            # Load models
            teacher_model = AutoModelForSequenceClassification.from_pretrained(
                teacher_name, 
                num_labels=self.num_labels
            ).to(self.device)
            
            student_model = AutoModelForSequenceClassification.from_pretrained(
                student_name, 
                num_labels=self.num_labels
            ).to(self.device)
            
            # Load tokenizer (we use the student's tokenizer)
            tokenizer = AutoTokenizer.from_pretrained(student_name)
            
            # Define training arguments
            student_output_dir = os.path.join(self.output_dir, f"{student_name.split('/')[-1]}_distilled")
            training_args = TrainingArguments(
                output_dir=student_output_dir,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                save_total_limit=1,
                load_best_model_at_end=True,
                evaluation_strategy="epoch",
                **training_kwargs
            )
            
            # Initialize distillation trainer
            trainer = DistillationTrainer(
                teacher_model=teacher_model,
                alpha=alpha,
                temperature=temperature,
                model=student_model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"] if "validation" in dataset else dataset["test"],
                tokenizer=tokenizer
            )
            
            # Train the student model
            trainer.train()
            
            # Evaluate the student model
            eval_results = trainer.evaluate()
            logger.info(f"Evaluation results for {student_name}: {eval_results}")
            
            # Save the student model
            trainer.save_model(student_output_dir)
            saved_models.append(student_output_dir)
            
            # Use the current student as the next teacher if needed
            if i < len(self.model_hierarchy) - 2:
                teacher_model = student_model
            
        return saved_models
            
    @classmethod
    def create_model_hierarchy(cls, task: str = "text-classification", num_levels: int = 3) -> List[str]:
        """
        Create a default model hierarchy based on the task.
        
        Args:
            task: Task type ('text-classification', 'question-answering', etc.)
            num_levels: Number of models in the hierarchy
            
        Returns:
            List of model names for the hierarchy
        """
        hierarchies = {
            "text-classification": [
                "roberta-large",
                "roberta-base",
                "distilroberta-base"
            ],
            "question-answering": [
                "deepset/roberta-large-squad2",
                "deepset/roberta-base-squad2",
                "distilbert-base-cased-distilled-squad"
            ],
            "language-generation": [
                "gpt2-medium",
                "gpt2",
                "distilgpt2"
            ]
        }
        
        if task not in hierarchies:
            raise ValueError(f"Unknown task: {task}. Supported tasks: {list(hierarchies.keys())}")
        
        # Return the default hierarchy, limited to the requested number of levels
        return hierarchies[task][:num_levels]
