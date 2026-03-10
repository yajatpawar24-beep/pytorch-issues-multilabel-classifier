"""
Preprocess PyTorch GitHub issues for multi-label classification.
"""

import ast
import json
import argparse
from pathlib import Path
from collections import Counter
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


def parse_and_extract_labels(label_str):
    """
    Parse stringified labels and extract label names.
    
    Args:
        label_str: String representation of label list
        
    Returns:
        List of label names
    """
    try:
        label_dicts = ast.literal_eval(label_str)
        return [label['name'] for label in label_dicts if isinstance(label, dict)]
    except:
        return []


def preprocess_and_save(
    data_path,
    output_dir,
    model_name="distilbert-base-uncased",
    min_label_count=100,
    test_size=0.2,
    seed=42
):
    """
    Complete preprocessing pipeline.
    
    Args:
        data_path: Path to raw JSONL data
        output_dir: Directory to save processed data
        model_name: Pretrained model name
        min_label_count: Minimum label occurrence threshold
        test_size: Test set fraction
        seed: Random seed
    """
    # Load the existing file
    df = pd.read_json(data_path, lines=True)
    print(f"Loaded {len(df)} issues")

    # Parse the stringified labels and extract names
    df['labels'] = df['labels'].apply(parse_and_extract_labels)

    # Convert to HF dataset
    dataset = Dataset.from_pandas(df)

    # Count all labels
    all_labels = []
    for issue in dataset:
        all_labels.extend(issue['labels'])

    label_counts = Counter(all_labels)
    print(f"Total unique labels: {len(label_counts)}")
    print(f"Most common: {label_counts.most_common(20)}")

    # Filter labels by minimum count
    valid_labels = [
        label for label, count in label_counts.items()
        if count >= min_label_count
    ]
    valid_labels.sort()

    print(f"\nUsing {len(valid_labels)} labels (out of {len(label_counts)})")
    print(f"Minimum count threshold: {min_label_count}")

    # Create mappings
    label2id = {label: idx for idx, label in enumerate(valid_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(valid_labels)

    print("\nFinal label set:")
    for label in valid_labels:
        print(f"  {label:40s} (count: {label_counts[label]})")

    # Keep only issues that have at least one valid label
    def has_valid_labels(example):
        return any(label in label2id for label in example['labels'])

    filtered_dataset = dataset.filter(has_valid_labels)

    print(f"\nDataset filtering:")
    print(f"  Original size: {len(dataset)}")
    print(f"  Filtered size: {len(filtered_dataset)}")
    print(f"  Removed: {len(dataset) - len(filtered_dataset)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        """Preprocess PyTorch GitHub issues for multi-label classification."""
        # Tokenize
        tokenized = tokenizer(
            examples['body'],
            truncation=True,
            return_tensors=None
        )

        # Convert labels to binary vectors
        binary_labels = []
        for issue_labels in examples['labels']:
            binary_vector = [0.0] * num_labels
            
            # Set 1.0 for each valid label present
            for label in issue_labels:
                if label in label2id:
                    idx = label2id[label]
                    binary_vector[idx] = 1.0

            binary_labels.append(binary_vector)

        tokenized['labels'] = binary_labels
        return tokenized

    # Test on a single example first
    print("\n--- Testing preprocessing on one example ---")
    test_example = filtered_dataset[0]
    print(f"Original labels: {test_example['labels']}")

    test_processed = preprocess_function({
        'title': [test_example['title']],
        'body': [test_example['body']],
        'labels': [test_example['labels']]
    })

    print(f"Binary vector: {test_processed['labels'][0]}")
    print(f"Which labels are 1: {[id2label[i] for i, val in enumerate(test_processed['labels'][0]) if val == 1.0]}")

    # Map the tokenization function to the filtered dataset
    tokenized_dataset = filtered_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1000,
        remove_columns=filtered_dataset.column_names,
        desc="Tokenizing dataset"
    )

    print(f"\nTokenized dataset:")
    print(tokenized_dataset)
    print(f"\nExample processed item keys: {tokenized_dataset[0].keys()}")
    print(f"Input IDs shape: {len(tokenized_dataset[0]['input_ids'])}")
    print(f"Labels shape: {len(tokenized_dataset[0]['labels'])}")

    # Create Train and Test Split
    tokenized_dataset = tokenized_dataset.train_test_split(
        test_size=test_size,
        seed=seed
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(tokenized_dataset['train'])} examples")
    print(f"  Test:  {len(tokenized_dataset['test'])} examples")

    # Save processed dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(str(output_path))
    print(f"\nSaved processed dataset to {output_path}")

    # Save label mappings
    label_mappings = {
        'label2id': label2id,
        'id2label': id2label,
        'num_labels': num_labels,
        'valid_labels': valid_labels
    }

    mappings_path = output_path / "label_mappings.json"
    with open(mappings_path, 'w') as f:
        json.dump(label_mappings, f, indent=2)
    
    print(f"Saved label mappings to {mappings_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PyTorch issues")
    parser.add_argument("--data-path", required=True, help="Path to JSONL file")
    parser.add_argument("--output-dir", default="./pytorch-issues-processed", help="Output directory")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Model name")
    parser.add_argument("--min-label-count", type=int, default=100, help="Minimum label count")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    preprocess_and_save(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        min_label_count=args.min_label_count,
        test_size=args.test_size,
        seed=args.seed
    )
