"""
Context-Sensitive Pruning module for HATD framework.
Analyzes input context to apply variable pruning ratios.
"""

import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Union, Optional, Callable
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK packages are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class QueryClassifier:
    """
    Classifies queries to determine their complexity level.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        label_map: Optional[Dict[int, str]] = None
    ):
        """
        Initialize the query classifier.
        
        Args:
            model_name: Pre-trained model to use for classification
            num_labels: Number of complexity labels (typically 3: simple, medium, complex)
            device: Device to run the model on
            label_map: Mapping from label indices to label names
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device
        
        # Default label map if none provided
        self.label_map = label_map or {
            0: "simple",   # Simple factual queries
            1: "medium",   # Medium complexity queries
            2: "complex"   # Complex reasoning queries
        }
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        ).to(device)
        self.model.eval()
        
    def classify(self, query: str) -> Dict[str, Union[str, float]]:
        """
        Classify the complexity of the input query.
        
        Args:
            query: Input query text
            
        Returns:
            Dictionary with 'label' and 'confidence' keys
        """
        # Tokenize input
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get predicted label and confidence
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        # Map numeric label to text label
        label = self.label_map[predicted_class]
        
        return {
            "label": label,
            "confidence": float(confidence),
            "label_id": int(predicted_class),
            "probabilities": {self.label_map[i]: float(prob) for i, prob in enumerate(probabilities)}
        }


class ContextSensitivePruner:
    """
    Applies variable pruning strategies based on input context.
    """
    
    def __init__(
        self,
        classifier: Optional[QueryClassifier] = None,
        default_pruning_ratio: float = 0.3,
        pruning_strategies: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the context-sensitive pruner.
        
        Args:
            classifier: QueryClassifier instance for determining query complexity
            default_pruning_ratio: Default ratio of tokens to prune when no specific strategy applies
            pruning_strategies: Mapping from query types to pruning ratios
        """
        self.classifier = classifier or QueryClassifier()
        
        # Default pruning strategies based on query complexity
        self.pruning_strategies = pruning_strategies or {
            "simple": 0.5,     # Aggressive pruning for simple queries
            "medium": 0.3,     # Moderate pruning for medium complexity
            "complex": 0.1     # Conservative pruning for complex reasoning
        }
        
        self.default_pruning_ratio = default_pruning_ratio
        
        # Rules for identifying query types based on keywords and patterns
        self.query_patterns = {
            "reasoning": re.compile(r'\b(why|how|explain|reason|because|therefore|if|then|solve|calculate)\b', re.IGNORECASE),
            "factual": re.compile(r'\b(what|when|where|who|which|define|list|name)\b', re.IGNORECASE),
            "generation": re.compile(r'\b(create|generate|write|compose|summarize|paraphrase)\b', re.IGNORECASE),
            "multi_step": re.compile(r'\b(steps?|first|second|third|next|then|finally)\b', re.IGNORECASE)
        }
        
    def detect_query_features(self, query: str) -> Dict[str, bool]:
        """
        Detect various features in the query text to guide pruning strategy.
        
        Args:
            query: Input query text
            
        Returns:
            Dictionary of features and their presence in the query
        """
        features = {}
        
        # Check for pattern matches
        for pattern_name, pattern in self.query_patterns.items():
            features[pattern_name] = bool(pattern.search(query))
            
        # Check for other textual features
        features["long_query"] = len(query.split()) > 30
        features["question"] = "?" in query
        features["multiple_sentences"] = len(sent_tokenize(query)) > 1
        
        return features
    
    def determine_pruning_ratio(self, query: str, features: Optional[Dict[str, bool]] = None) -> float:
        """
        Determine the appropriate pruning ratio based on query classification and features.
        
        Args:
            query: Input query text
            features: Pre-computed query features (if available)
            
        Returns:
            Pruning ratio between 0 and 1
        """
        # Detect features if not provided
        if features is None:
            features = self.detect_query_features(query)
            
        # Classify query complexity
        classification = self.classifier.classify(query)
        complexity = classification["label"]
        
        # Base pruning ratio from complexity classification
        pruning_ratio = self.pruning_strategies.get(complexity, self.default_pruning_ratio)
        
        # Adjust based on specific features
        if features.get("reasoning", False) or features.get("multi_step", False):
            # Reduce pruning for reasoning tasks to preserve intermediate steps
            pruning_ratio *= 0.7
        
        if features.get("long_query", False):
            # More aggressive pruning for long queries
            pruning_ratio *= 1.2
            
        # Ensure pruning_ratio is in valid range [0, 1]
        pruning_ratio = max(0.0, min(0.9, pruning_ratio))
        
        return pruning_ratio
    
    def prune_text(
        self, 
        text: str, 
        pruning_function: Callable[[str, float], str],
        pruning_ratio: Optional[float] = None
    ) -> str:
        """
        Apply context-sensitive pruning to the input text.
        
        Args:
            text: Input text to prune
            pruning_function: Function that takes (text, pruning_ratio) and returns pruned text
            pruning_ratio: Override the automatically determined pruning ratio
            
        Returns:
            Pruned text
        """
        # Determine pruning ratio if not provided
        if pruning_ratio is None:
            pruning_ratio = self.determine_pruning_ratio(text)
            
        # Apply pruning function with the determined ratio
        pruned_text = pruning_function(text, pruning_ratio)
        
        return pruned_text


class RuleBased_Pruner:
    """
    Implements rule-based token pruning strategies.
    """
    
    def __init__(self):
        """
        Initialize the rule-based pruner with common stopwords and redundant phrases.
        """
        # Common stopwords that can often be pruned
        self.stopwords = set([
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
            "by", "about", "as", "of", "that", "this", "these", "those", "such", "so",
            "very", "quite", "just", "really", "simply", "basically", "actually", "definitely"
        ])
        
        # Redundant phrases that can be pruned
        self.redundant_phrases = [
            r'\bin my opinion\b',
            r'\bi think\b',
            r'\bplease note that\b',
            r'\bas you can see\b',
            r'\bin other words\b',
            r'\bit is important to realize\b',
            r'\bthe fact that\b',
            r'\bneedless to say\b',
            r'\bit goes without saying\b'
        ]
        
        # Compile redundant phrase patterns
        self.redundant_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.redundant_phrases]
        
    def prune(self, text: str, pruning_ratio: float = 0.3) -> str:
        """
        Apply rule-based pruning to the input text.
        
        Args:
            text: Input text to prune
            pruning_ratio: Ratio of tokens to attempt to prune
            
        Returns:
            Pruned text
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # Remove redundant phrases
        for i, sentence in enumerate(sentences):
            for pattern in self.redundant_patterns:
                sentence = pattern.sub('', sentence)
            sentences[i] = sentence
        
        # Calculate how many words to retain
        words = []
        for sentence in sentences:
            words.extend(sentence.split())
        
        total_words = len(words)
        words_to_keep = int(total_words * (1 - pruning_ratio))
        
        # Prioritize words to retain (non-stopwords)
        word_priorities = []
        for i, word in enumerate(words):
            # Lower priority (more likely to be pruned) if it's a stopword
            priority = 1 if word.lower() not in self.stopwords else 0
            word_priorities.append((i, word, priority))
        
        # Sort by priority (higher priority words kept)
        word_priorities.sort(key=lambda x: x[2], reverse=True)
        
        # Keep the top words_to_keep words by priority
        indices_to_keep = sorted([item[0] for item in word_priorities[:words_to_keep]])
        
        # Reconstruct text with only the kept words
        kept_words = [words[i] for i in indices_to_keep]
        pruned_text = ' '.join(kept_words)
        
        return pruned_text
