"""
Token-Aware Compression module for HATD framework.
Identifies high-value tokens using attention weights or gradient-based saliency scores.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Union, Optional


class TokenScorer:
    """
    Scores tokens based on their importance to the model's output.
    """
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        scoring_method: str = "attention",
        attention_layer: int = -1,
        attention_head: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the token scorer.
        
        Args:
            model_name: Name of the model to use for scoring
            scoring_method: Method to use for scoring tokens ('attention' or 'gradient')
            attention_layer: Which layer to extract attention from (-1 for last layer)
            attention_head: Which attention head to use (None for average across heads)
            device: Device to run the model on
        """
        self.model_name = model_name
        self.scoring_method = scoring_method
        self.attention_layer = attention_layer
        self.attention_head = attention_head
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
        self.model.eval()
        
    def score_tokens(self, text: str) -> Tuple[List[str], List[float]]:
        """
        Score tokens in the input text based on the chosen scoring method.
        
        Args:
            text: Input text to score
            
        Returns:
            Tuple of (tokens, scores)
        """
        if self.scoring_method == "attention":
            return self._score_with_attention(text)
        elif self.scoring_method == "gradient":
            return self._score_with_gradient(text)
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")
    
    def _score_with_attention(self, text: str) -> Tuple[List[str], List[float]]:
        """
        Score tokens using attention weights.
        
        Args:
            text: Input text to score
            
        Returns:
            Tuple of (tokens, scores)
        """
       
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        attentions = outputs.attentions[self.attention_layer]
        if self.attention_head is None:
            attention_scores = attentions.mean(dim=1).squeeze(0)
        else:
            attention_scores = attentions[:, self.attention_head].squeeze(0)
        
        token_importance = attention_scores.sum(dim=0).cpu().numpy()
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        return tokens, token_importance.tolist()
    
    def _score_with_gradient(self, text: str) -> Tuple[List[str], List[float]]:
        """
        Score tokens using gradient-based saliency.
        
        Args:
            text: Input text to score
            
        Returns:
            Tuple of (tokens, scores)
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Enable gradient calculation
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Clear previous gradients
        self.model.zero_grad()
        
        # Get embeddings with gradient tracking
        embeddings = self.model.get_input_embeddings()(inputs.input_ids)
        embeddings.retain_grad()
        
        # Forward pass
        outputs = self.model(inputs_embeds=embeddings)  
        # Use the mean of the last hidden state as the output to compute gradients
        output = outputs.last_hidden_state.mean()
        
        # Backward pass
        output.backward()
        
        grad_magnitude = torch.norm(embeddings.grad, dim=2).squeeze(0).detach().cpu().numpy()
        
        # Decode tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        return tokens, grad_magnitude.tolist()


class TokenCompressor:
    """
    Compresses input by removing low-importance tokens.
    """
    
    def __init__(
        self, 
        scorer: TokenScorer,
        compression_ratio: float = 0.3,
        preserve_special_tokens: bool = True,
        preserve_sentence_structure: bool = True
    ):
        """
        Initialize the token compressor.
        
        Args:
            scorer: TokenScorer instance to score tokens
            compression_ratio: Target ratio of tokens to remove (0-1)
            preserve_special_tokens: Whether to always keep special tokens
            preserve_sentence_structure: Whether to preserve basic sentence structure
        """
        self.scorer = scorer
        self.compression_ratio = compression_ratio
        self.preserve_special_tokens = preserve_special_tokens
        self.preserve_sentence_structure = preserve_sentence_structure
        self.tokenizer = scorer.tokenizer
        
        # Define tokens to always preserve if preserve_sentence_structure is True
        self.structural_tokens = set([
            ".", ",", "!", "?", ":", ";",
            "the", "a", "an", "is", "are", "was", "were",
            "and", "or", "but", "if", "then", "because", "so"
        ])
        
    def compress(self, text: str) -> str:
        """
        Compress the input text by removing least important tokens.
        
        Args:
            text: Input text to compress
            
        Returns:
            Compressed text
        """
        tokens, scores = self.scorer.score_tokens(text)
        token_scores = list(zip(tokens, scores))
        num_tokens = len(tokens)
        num_to_remove = int(num_tokens * self.compression_ratio)
        
        if self.preserve_special_tokens or self.preserve_sentence_structure:
            filtered_token_scores = []
            for token, score in token_scores:
                if (self.preserve_special_tokens and token in self.tokenizer.all_special_tokens) or \
                   (self.preserve_sentence_structure and token.lower() in self.structural_tokens):
                    filtered_token_scores.append((token, float('inf')))
                else:
                    filtered_token_scores.append((token, score))
            token_scores = filtered_token_scores
        sorted_token_scores = sorted(token_scores, key=lambda x: x[1])
        tokens_to_remove = set(token for token, _ in sorted_token_scores[:num_to_remove])
        preserved_tokens = [token for token in tokens if token not in tokens_to_remove]
        compressed_text = ""
        i = 0
        while i < len(preserved_tokens):
            token = preserved_tokens[i]
            if token.startswith("##"):
                compressed_text = compressed_text.rstrip() + token[2:]
            else:
                if i > 0:
                    compressed_text += " "
                compressed_text += token
            i += 1
            
        return compressed_text
