import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re

class TextDataset(Dataset):
    """Dataset class for character-level text data."""
    
    def __init__(self, text, char_to_idx, sequence_length):
        self.text = text
        self.char_to_idx = char_to_idx
        self.sequence_length = sequence_length
        
        # Convert text to indices
        self.data = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in text]
        
        # Create sequences
        self.sequences = []
        for i in range(len(self.data) - sequence_length):
            self.sequences.append((
                self.data[i:i + sequence_length],
                self.data[i + 1:i + sequence_length + 1]
            ))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

class TextDataLoader:
    """Data loader for text data preprocessing and batching."""
    
    def __init__(self, text, sequence_length=50, batch_size=16, min_freq=1):
        self.text = self.preprocess_text(text)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.min_freq = min_freq
        
        # Build vocabulary
        self.build_vocabulary()
        
        # Create dataset
        self.dataset = TextDataset(self.text, self.char_to_idx, sequence_length)
        
        # Statistics
        self.vocab_size = len(self.char_to_idx)
        self.total_chars = len(self.text)
        self.num_sequences = len(self.dataset)
    
    def preprocess_text(self, text):
        """Preprocess the input text."""
        # Basic cleaning
        text = text.strip()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove or replace special characters if needed
        # For educational purposes, we'll keep most characters
        
        return text
    
    def build_vocabulary(self):
        """Build character-level vocabulary."""
        # Count character frequencies
        char_counts = Counter(self.text)
        
        # Filter characters by minimum frequency
        filtered_chars = [char for char, count in char_counts.items() if count >= self.min_freq]
        
        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        vocab_chars = special_tokens + sorted(list(set(filtered_chars)))
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Store vocabulary statistics
        self.char_frequencies = char_counts
        self.unique_chars = len(set(self.text))
        self.vocab_coverage = len(filtered_chars) / self.unique_chars if self.unique_chars > 0 else 0
    
    def get_data_loader(self, shuffle=True):
        """Get PyTorch DataLoader."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def encode_text(self, text):
        """Encode text to indices."""
        return [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in text]
    
    def decode_indices(self, indices):
        """Decode indices to text."""
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in indices])
    
    def get_char_distribution(self):
        """Get character distribution statistics."""
        total_chars = sum(self.char_frequencies.values())
        
        distribution = {}
        for char, count in self.char_frequencies.most_common():
            if char in self.char_to_idx:
                distribution[char] = {
                    'count': count,
                    'frequency': count / total_chars,
                    'index': self.char_to_idx[char]
                }
        
        return distribution
    
    def get_vocabulary_info(self):
        """Get vocabulary information."""
        return {
            'vocab_size': self.vocab_size,
            'total_chars': self.total_chars,
            'unique_chars': self.unique_chars,
            'vocab_coverage': self.vocab_coverage,
            'num_sequences': self.num_sequences,
            'sequence_length': self.sequence_length,
            'batch_size': self.batch_size
        }
    
    def get_sample_sequences(self, num_samples=5):
        """Get sample input-target sequence pairs."""
        samples = []
        
        for i in range(min(num_samples, len(self.dataset))):
            input_seq, target_seq = self.dataset[i]
            
            input_text = self.decode_indices(input_seq.tolist())
            target_text = self.decode_indices(target_seq.tolist())
            
            samples.append({
                'input': input_text,
                'target': target_text,
                'input_indices': input_seq.tolist(),
                'target_indices': target_seq.tolist()
            })
        
        return samples
    
    def split_data(self, train_ratio=0.8, val_ratio=0.1):
        """Split data into train, validation, and test sets."""
        total_size = len(self.dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Create random split
        indices = torch.randperm(total_size)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        val_dataset = torch.utils.data.Subset(self.dataset, val_indices)
        test_dataset = torch.utils.data.Subset(self.dataset, test_indices)
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, train_ratio=0.8, val_ratio=0.1):
        """Create train, validation, and test data loaders."""
        train_dataset, val_dataset, test_dataset = self.split_data(train_ratio, val_ratio)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader, test_loader
