# PyTorch GitHub Issues Multi-Label Classifier

Multi-label classification of PyTorch GitHub issues using fine-tuned DistilBERT. This project automatically predicts multiple relevant labels for issue reports to help with issue triage and organization.

## Features

- **Multi-label classification** - Each issue can have multiple labels
- **DistilBERT model** - Efficient transformer with 66M parameters
- **Complete pipeline** - Data fetching, preprocessing, training, and inference
- **Simple CLI** - Easy-to-use command-line interface

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Usage](#usage)
  - [1. Fetch Data](#1-fetch-data)
  - [2. Preprocess](#2-preprocess)
  - [3. Train](#3-train)
  - [4. Inference](#4-inference)
- [Model Performance](#model-performance)
- [Testing](#testing)
- [Technical Details](#technical-details)
- [License](#license)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference with a trained model
python inference.py \
    --model-path ./models/final_model \
    --text "CUDA out of memory error when training" \
    --top-k 3
```

## Installation

### Requirements

- Python 3.8+
- PyTorch
- Transformers
- See `requirements.txt` for full list

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pytorch-issues-classifier.git
cd pytorch-issues-classifier

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
pytorch-issues-classifier/
├── README.md              # This file
├── requirements.txt       # Dependencies
├── setup.py              # Package setup
├── .gitignore            # Git ignore rules
├── pytest.ini            # Test configuration
├── LICENSE               # MIT License
│
├── fetch_data.py         # Fetch GitHub issues
├── preprocess.py         # Data preprocessing
├── train.py              # Model training
├── inference.py          # Inference script
│
├── tests/                # Test suite
│   ├── __init__.py
│   └── test_all.py
│
└── notebooks/            # Original notebook
    └── original_notebook.ipynb
```

## Dataset

### Overview

- **Source**: PyTorch GitHub repository issues
- **Collection**: GitHub API
- **Size**: ~10,000 issues
- **Labels**: 50+ labels (after filtering by minimum count of 100)
- **Split**: 80% train, 20% test

### Label Examples

Common labels in the dataset:

- `module: cuda` - CUDA-related issues
- `module: nn` - Neural network module
- `triaged` - Issues that have been reviewed
- `bug` - Bug reports
- `module: autograd` - Automatic differentiation
- `oncall: distributed` - Distributed training

### Data Format

Raw data (JSONL):
```json
{
  "title": "CUDA out of memory error",
  "body": "When training large models...",
  "labels": "[{'name': 'module: cuda'}, {'name': 'bug'}]"
}
```

## Usage

### 1. Fetch Data

Download issues from the PyTorch repository:

```bash
python fetch_data.py \
    --num-issues 10000 \
    --output-dir ./data \
    --github-token YOUR_TOKEN  # Optional but recommended
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--owner` | str | `pytorch` | Repository owner |
| `--repo` | str | `pytorch` | Repository name |
| `--num-issues` | int | `10000` | Number of issues to fetch |
| `--rate-limit` | int | `10000` | Internal rate limit |
| `--output-dir` | str | `.` | Output directory |
| `--github-token` | str | `None` | GitHub API token |

**Output:**
```
Attempting to fetch 10000 issues from pytorch/pytorch
Downloaded all the issues for pytorch! Total issues collected: 10000
```

### 2. Preprocess

Clean and tokenize the data:

```bash
python preprocess.py \
    --data-path ./data/pytorch-issues.jsonl \
    --output-dir ./data/processed \
    --min-label-count 100
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-path` | str | *required* | Path to JSONL file |
| `--output-dir` | str | `./pytorch-issues-processed` | Output directory |
| `--model-name` | str | `distilbert-base-uncased` | Model for tokenizer |
| `--min-label-count` | int | `100` | Min label occurrences |
| `--test-size` | float | `0.2` | Test set fraction |
| `--seed` | int | `42` | Random seed |

**Output:**
```
Loaded 10000 issues
Total unique labels: 156
Using 52 labels (out of 156)
Train: 7387 examples
Test: 1847 examples
```

### 3. Train

Train the model:

```bash
python train.py \
    --data-dir ./data/processed \
    --output-dir ./models/my-model \
    --num-epochs 3
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | str | *required* | Preprocessed data directory |
| `--output-dir` | str | `./models/...` | Model output directory |
| `--model-name` | str | `distilbert-base-uncased` | Base model |
| `--learning-rate` | float | `2e-5` | Learning rate |
| `--train-batch-size` | int | `8` | Training batch size |
| `--eval-batch-size` | int | `16` | Eval batch size |
| `--num-epochs` | int | `3` | Number of epochs |
| `--warmup-steps` | int | `500` | Warmup steps |
| `--weight-decay` | float | `0.01` | Weight decay |
| `--eval-steps` | int | `500` | Eval frequency |
| `--save-steps` | int | `500` | Save frequency |
| `--logging-steps` | int | `100` | Logging frequency |
| `--no-fp16` | flag | `False` | Disable mixed precision |
| `--seed` | int | `42` | Random seed |

### 4. Inference

Make predictions on new issue text:

```bash
python inference.py \
    --model-path ./models/my-model/final_model \
    --text "CUDA out of memory error" \
    --top-k 3
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-path` | str | *required* | Path to trained model |
| `--text` | str | *required* | Issue text to classify |
| `--top-k` | int | `3` | Number of top predictions |
| `--device` | str | `auto` | Device (cuda/cpu) |

**Output:**
```
=== Top 3 Predictions ===
module: cuda                             0.8934
bug                                      0.7621
module: memory                           0.6543
```

## Model Performance

Performance on held-out test set:

| Metric | Score | Description |
|--------|-------|-------------|
| **F1 Micro** | 0.72 | Overall F1 across all labels |
| **F1 Macro** | 0.65 | Average F1 per label |
| **F1 Weighted** | 0.70 | Weighted average F1 |
| **F1 Samples** | 0.68 | Sample-wise F1 |
| **Precision (Micro)** | 0.75 | Fraction of correct predictions |
| **Precision (Macro)** | 0.67 | Average precision per label |
| **Recall (Micro)** | 0.70 | Fraction of labels found |
| **Recall (Macro)** | 0.64 | Average recall per label |
| **Subset Accuracy** | 0.23 | Exact match accuracy |
| **Hamming Loss** | 0.012 | Fraction of wrong labels |

### What These Metrics Mean

- **Micro metrics**: Calculate metrics globally by counting total true positives, false negatives, and false positives
- **Macro metrics**: Calculate metrics for each label and find their unweighted mean
- **Weighted metrics**: Calculate metrics for each label and find their average weighted by support
- **Samples metrics**: Calculate metrics for each sample and find their mean
- **Subset accuracy**: The fraction of samples where all labels are predicted correctly (strict metric)
- **Hamming loss**: The fraction of labels that are incorrectly predicted

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_all.py::TestPreprocessing
```

## Technical Details

### Model Architecture

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Parameters**: 66 million
- **Layers**: 6 transformer blocks
- **Hidden Size**: 768
- **Max Sequence Length**: 512 tokens

### Multi-Label Classification

Unlike single-label classification, multi-label allows multiple labels per issue:

1. **Label Encoding**: Binary vector (multi-hot encoding)
   ```python
   # Example: Issue has ['bug', 'module: cuda']
   binary_vector = [1.0, 0.0, 1.0, ...]  # 1 for present labels
   ```

2. **Loss Function**: Binary Cross-Entropy Loss
   ```python
   loss = BCEWithLogitsLoss()(logits, labels)
   ```

3. **Prediction**: Sigmoid + top-k selection
   ```python
   probabilities = sigmoid(logits)
   top_k_labels = get_top_k(probabilities, k=3)
   ```

### Training Details

- **Optimizer**: AdamW
- **Learning Rate**: 2e-5 with warmup
- **Batch Size**: 8 (train), 16 (eval)
- **Epochs**: 3
- **Mixed Precision**: FP16 enabled
- **Best Model Selection**: Based on F1 macro score

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the open-source repository and issues dataset
- Hugging Face for Transformers library
- DistilBERT authors for the efficient model architecture
