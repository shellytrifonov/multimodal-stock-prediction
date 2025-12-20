import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTrainingDataShapes(unittest.TestCase):
    """Unit tests for LSTM training data tensor shapes."""
    
    def test_twitter_shapes(self):
        """Test Twitter LSTM training data shape (samples, 72, 5)."""
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'twitter_lstm_training_data.npz')
        
        if not os.path.exists(data_path):
            self.skipTest(f"Training data not found: {data_path}")
        
        print(f"\n{'='*70}")
        print("Testing Twitter LSTM Training Data")
        print('='*70)
        
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        
        print(f"\nLoaded from: {data_path}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        self.assertEqual(len(X.shape), 3, 
                        f"X should be 3D tensor, got {len(X.shape)}D")
        self.assertEqual(X.shape[1], 72, 
                        f"Expected 72 timesteps, got {X.shape[1]}")
        self.assertEqual(X.shape[2], 5, 
                        f"Expected 5 features, got {X.shape[2]}")
        self.assertEqual(len(y.shape), 1, 
                        f"y should be 1D, got {len(y.shape)}D")
        self.assertEqual(len(X), len(y), 
                        f"X and y must have same sample count: {len(X)} vs {len(y)}")
        self.assertTrue(np.issubdtype(X.dtype, np.floating), 
                       f"X should be float type, got {X.dtype}")
        self.assertTrue(np.issubdtype(y.dtype, np.integer), 
                       f"y should be integer type, got {y.dtype}")
        unique_labels = np.unique(y)
        self.assertTrue(set(unique_labels).issubset({0, 1}), 
                       f"Labels should be binary (0 or 1), got {unique_labels}")
        
        print(f"\n✓ All Twitter LSTM shape tests passed")
        print(f"  - Samples: {X.shape[0]}")
        print(f"  - Timesteps: {X.shape[1]}")
        print(f"  - Features: {X.shape[2]}")
        print(f"  - Label distribution: {np.bincount(y)}")
    
    def test_news_shapes(self):
        """Test News LSTM training data shape (samples, 72, 8)."""
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'news_lstm_training_data.npz')
        
        if not os.path.exists(data_path):
            self.skipTest(f"Training data not found: {data_path}")
        
        print(f"\n{'='*70}")
        print("Testing News LSTM Training Data")
        print('='*70)
        
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        
        print(f"\nLoaded from: {data_path}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        self.assertEqual(len(X.shape), 3, 
                        f"X should be 3D tensor, got {len(X.shape)}D")
        self.assertEqual(X.shape[1], 72, 
                        f"Expected 72 timesteps, got {X.shape[1]}")
        self.assertEqual(X.shape[2], 8, 
                        f"Expected 8 features, got {X.shape[2]}")
        self.assertEqual(len(y.shape), 1, 
                        f"y should be 1D, got {len(y.shape)}D")
        self.assertEqual(len(X), len(y), 
                        f"X and y must have same sample count: {len(X)} vs {len(y)}")
        self.assertTrue(np.issubdtype(X.dtype, np.floating), 
                       f"X should be float type, got {X.dtype}")
        self.assertTrue(np.issubdtype(y.dtype, np.integer), 
                       f"y should be integer type, got {y.dtype}")
        unique_labels = np.unique(y)
        self.assertTrue(set(unique_labels).issubset({0, 1}), 
                       f"Labels should be binary (0 or 1), got {unique_labels}")
        
        print(f"\n✓ All News LSTM shape tests passed")
        print(f"  - Samples: {X.shape[0]}")
        print(f"  - Timesteps: {X.shape[1]}")
        print(f"  - Features: {X.shape[2]}")
        print(f"  - Label distribution: {np.bincount(y)}")
    
    def test_feature_statistics(self):
        """Test basic statistical properties of feature data."""
        twitter_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'twitter_lstm_training_data.npz')
        news_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'news_lstm_training_data.npz')
        
        print(f"\n{'='*70}")
        print("Testing Feature Statistics")
        print('='*70)
        
        if os.path.exists(twitter_path):
            data = np.load(twitter_path)
            X = data['X']
            
            print(f"\nTwitter Features:")
            self.assertFalse(np.isnan(X).any(), "Twitter X contains NaN values")
            self.assertFalse(np.isinf(X).any(), "Twitter X contains Inf values")
            print(f"  ✓ No NaN or Inf values")
            print(f"  - Mean: {X.mean():.4f}")
            print(f"  - Std: {X.std():.4f}")
            print(f"  - Min: {X.min():.4f}")
            print(f"  - Max: {X.max():.4f}")
        
        if os.path.exists(news_path):
            data = np.load(news_path)
            X = data['X']
            
            print(f"\nNews Features:")
            self.assertFalse(np.isnan(X).any(), "News X contains NaN values")
            self.assertFalse(np.isinf(X).any(), "News X contains Inf values")
            print(f"  ✓ No NaN or Inf values")
            print(f"  - Mean: {X.mean():.4f}")
            print(f"  - Std: {X.std():.4f}")
            print(f"  - Min: {X.min():.4f}")
            print(f"  - Max: {X.max():.4f}")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("LSTM Training Data Shape Verification Tests")
    print("=" * 70)
    print("\nRunning unit tests...")
    
    unittest.main(verbosity=2)
