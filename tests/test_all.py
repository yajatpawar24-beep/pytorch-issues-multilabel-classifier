"""
Tests for PyTorch GitHub Issues Classifier.
"""

import unittest
import tempfile
from pathlib import Path
from preprocess import parse_and_extract_labels


class TestPreprocessing(unittest.TestCase):
    """Test data preprocessing functions."""
    
    def test_parse_valid_labels(self):
        """Test parsing valid label strings."""
        label_str = "[{'name': 'bug'}, {'name': 'feature'}]"
        result = parse_and_extract_labels(label_str)
        self.assertEqual(result, ['bug', 'feature'])
    
    def test_parse_empty_labels(self):
        """Test parsing empty labels."""
        label_str = "[]"
        result = parse_and_extract_labels(label_str)
        self.assertEqual(result, [])
    
    def test_parse_invalid_labels(self):
        """Test parsing invalid label strings."""
        label_str = "invalid"
        result = parse_and_extract_labels(label_str)
        self.assertEqual(result, [])


class TestInference(unittest.TestCase):
    """Test inference functionality."""
    
    def test_predict_top_k_format(self):
        """Test that predict_top_k returns correct format."""
        from inference import predict_top_k
        import torch
        from unittest.mock import MagicMock
        
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock model output
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[2.0, -2.0, 0.5]])
        mock_model.return_value = mock_output
        
        id2label = {0: 'bug', 1: 'feature', 2: 'enhancement'}
        device = torch.device('cpu')
        
        predictions = predict_top_k(
            mock_model, mock_tokenizer, id2label, device, "test", k=2
        )
        
        # Should return 2 predictions
        self.assertEqual(len(predictions), 2)
        
        # Each should be (label, prob) tuple
        for label, prob in predictions:
            self.assertIsInstance(label, str)
            self.assertIsInstance(prob, float)


if __name__ == "__main__":
    unittest.main()
