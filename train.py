"""
Train PyTorch GitHub Issues Multi-Label Classifier.
"""

import json
import argparse
from pathlib import Path
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    hamming_loss
)


def compute_metrics(eval_pred):
    """
    Compute multi-label classification metrics.
    """
    logits, labels = eval_pred

    predictions = torch.sigmoid(torch.tensor(logits)).numpy()

    # Apply threshold to get binary predictions
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)

    precision_micro = precision_score(labels, binary_predictions, average='micro', zero_division=0)
    recall_micro = recall_score(labels, binary_predictions, average='micro', zero_division=0)
    f1_micro = f1_score(labels, binary_predictions, average='micro', zero_division=0)

    precision_macro = precision_score(labels, binary_predictions, average='macro', zero_division=0)
    recall_macro = recall_score(labels, binary_predictions, average='macro', zero_division=0)
    f1_macro = f1_score(labels, binary_predictions, average='macro', zero_division=0)

    f1_weighted = f1_score(labels, binary_predictions, average='weighted', zero_division=0)
    f1_samples = f1_score(labels, binary_predictions, average='samples', zero_division=0)
    subset_accuracy = accuracy_score(labels, binary_predictions)
    hamming = hamming_loss(labels, binary_predictions)

    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_samples': f1_samples,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'subset_accuracy': subset_accuracy,
        'hamming_loss': hamming,
    }


def train_model(
    data_dir,
    output_dir,
    model_name="distilbert-base-uncased",
    learning_rate=2e-5,
    train_batch_size=8,
    eval_batch_size=16,
    num_epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    fp16=True,
    seed=42
):
    """
    Train the multi-label classifier.
    
    Args:
        data_dir: Directory with preprocessed data
        output_dir: Directory to save model
        model_name: Base model name
        learning_rate: Learning rate
        train_batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        num_epochs: Number of epochs
        warmup_steps: Warmup steps
        weight_decay: Weight decay
        eval_steps: Evaluation frequency
        save_steps: Save frequency
        logging_steps: Logging frequency
        fp16: Use mixed precision
        seed: Random seed
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset
    print(f"Loading dataset from {data_dir}")
    tokenized_dataset = load_from_disk(data_dir)
    print(f"Train: {len(tokenized_dataset['train'])} examples")
    print(f"Test: {len(tokenized_dataset['test'])} examples")

    # Load label mappings
    mappings_path = Path(data_dir) / "label_mappings.json"
    with open(mappings_path, 'r') as f:
        label_info = json.load(f)

    label2id = label_info['label2id']
    id2label = label_info['id2label']
    num_labels = label_info['num_labels']

    print(f"Number of labels: {num_labels}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        
        # Training hyperparameters
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        
        # Optimization
        warmup_steps=warmup_steps,
        
        # Logging
        logging_dir=f'{output_dir}/logs',
        logging_steps=logging_steps,
        
        # Model selection
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        
        save_total_limit=2,
        fp16=fp16,
        report_to="none",
        seed=seed,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    final_model_path = Path(output_dir) / "final_model"
    print(f"\nSaving final model to {final_model_path}")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Copy label mappings
    import shutil
    shutil.copy(mappings_path, final_model_path / "label_mappings.json")

    # Final evaluation
    print("\nFinal evaluation:")
    results = trainer.evaluate()
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch issues classifier")
    parser.add_argument("--data-dir", required=True, help="Preprocessed data directory")
    parser.add_argument("--output-dir", default="./models/distilbert-pytorch-classifier", help="Output directory")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Model name")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Eval batch size")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--eval-steps", type=int, default=500, help="Eval steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save steps")
    parser.add_argument("--logging-steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=not args.no_fp16,
        seed=args.seed
    )
