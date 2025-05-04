# Project Sputnik

A Hierarchical Adaptive Token Distillation Test Bed (HATD)


This repository implements the HATD framework as described in "Hierarchical Adaptive Token Distillation: Optimizing Token Usage in Large Language Models" (May 2025).

## Overview

HATD addresses the high token consumption challenge in Large Language Models through three integrated mechanisms:

1. **Token-Aware Compression**: Identifies high-value tokens using attention weights or gradient-based saliency scores
2. **Hierarchical Distillation Pipeline**: Knowledge transfer from large teacher models to progressively smaller student models
3. **Context-Sensitive Pruning**: Analyzes input context to apply variable pruning ratios

## Project Structure

```
hatd-implementation/
├── configs/         # Configuration files for experiments
├── data/            # Datasets and data processing scripts
├── experiments/     # Experiment scripts and results
├── models/          # Model definitions and weights
├── src/             # Core implementation
│   ├── compression/ # Token-aware compression implementation
│   ├── distillation/# Hierarchical distillation implementation
│   ├── pruning/     # Context-sensitive pruning implementation
│   └── pipeline/    # Full HATD pipeline
├── utils/           # Utility functions and helpers
└── README.md        # Project documentation
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start Example

1. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the pipeline on a sample text file:
   ```bash
   python main.py --config configs/default_config.json --input data/sample_text.txt --output data/processed_text.md
   ```

3. The results will be saved in `data/processed_text.md`.

### What Happens- 5/4/2025
- The pipeline reads your input text, prunes and compresses tokens, and routes it to the right model based on complexity.
- Example result from our last test:
  - **Original tokens:** 253
  - **Processed tokens:** 97
  - **Token reduction:** 61.66%
  - **Selected model:** distilroberta-base
  - **Processing time:** ~1.6 seconds
  - **Output:** `data/processed_text.md`

---

## Project Status
- Runs end-to-end on local text files
- Handles NLTK dependency issues (see troubleshooting below)
- Model weights may show a warning if not fine-tuned, but pipeline works for testing

## Troubleshooting
If you see errors about missing NLTK data, run:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Evaluation

The framework is evaluated on multiple tasks:
- Question Answering (SQuAD)
- Text Generation (CommonGen)
- Reasoning (GSM8K)

## Results

As demonstrated in the paper, HATD end goal is to at some point:
- Reduce token usage by 30-60%
- Preserve over 95% of performance quality
- Decrease inference time proportionally to token reduction

This is all a pipe-dream, this is only just expermininty with this type of technology, these are strech goals set.

## License

MIT
