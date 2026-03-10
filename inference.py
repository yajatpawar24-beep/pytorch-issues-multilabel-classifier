"""
Inference for PyTorch GitHub Issues Classifier.
"""

import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def predict_top_k(model, tokenizer, id2label, device, body, k=3):
    """
    Return top K most likely labels regardless of threshold.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        id2label: Label ID to name mapping
        device: Device (cpu/cuda)
        body: Issue text
        k: Number of top predictions
        
    Returns:
        List of (label, probability) tuples
    """
    text = body if body else ""
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    # Get top K
    top_indices = np.argsort(probs)[-k:][::-1]
    predictions = [(id2label[i], float(probs[i])) for i in top_indices]

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Inference on PyTorch issues")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--text", required=True, help="Issue text")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load label mappings
    label_mappings_path = f"{args.model_path}/label_mappings.json"
    with open(label_mappings_path, 'r') as f:
        label_info = json.load(f)
        id2label = {int(k): v for k, v in label_info['id2label'].items()}

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()

    # Predict
    predictions = predict_top_k(model, tokenizer, id2label, device, args.text, args.top_k)

    print(f"\n=== Top {args.top_k} Predictions ===")
    for label, prob in predictions:
        print(f"{label:40s} {prob:.4f}")


if __name__ == "__main__":
    main()
