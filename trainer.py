import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

class ModelTrainer:
    """Trainer class for the language model."""
    
    def __init__(self, model, data_loader, learning_rate=0.003, device='cpu'):
        if isinstance(device, torch.device):
            device = str(device)
        self.model = model
        self.data_loader = data_loader
        self.device = device
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Training statistics
        self.training_losses = []
        self.current_epoch = 0
    
    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Get data loader
        train_loader = self.data_loader.get_data_loader()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Reshape for loss calculation
            # outputs: (batch_size, seq_len, vocab_size)
            # targets: (batch_size, seq_len)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Update learning rate scheduler
        self.scheduler.step(avg_loss)
        
        # Store training statistics
        self.training_losses.append(avg_loss)
        self.current_epoch += 1
        
        return avg_loss
    
    def evaluate(self, data_loader=None):
        """Evaluate the model on given data."""
        if data_loader is None:
            data_loader = self.data_loader.get_data_loader()
        
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Reshape for loss calculation
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = targets.reshape(-1)
                
                # Calculate loss
                loss = self.criterion(outputs_flat, targets_flat)
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs_flat, dim=1)
                correct = (predictions == targets_flat).sum().item()
                total_correct += correct
                total_tokens += targets_flat.size(0)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, accuracy
    
    def calculate_perplexity(self, data_loader=None):
        """Calculate perplexity of the model."""
        avg_loss, _ = self.evaluate(data_loader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        return perplexity.item()
    
    def save_checkpoint(self, filepath):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_losses': self.training_losses,
            'vocab_size': self.model.vocab_size,
            'embed_dim': self.model.embed_dim,
            'num_heads': self.model.num_heads,
            'num_layers': self.model.num_layers,
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_losses = checkpoint['training_losses']
        self.current_epoch = checkpoint['epoch']
        
        return checkpoint
    
    def get_learning_rate(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_training_stats(self):
        """Get training statistics."""
        return {
            'epoch': self.current_epoch,
            'losses': self.training_losses,
            'current_lr': self.get_learning_rate(),
            'best_loss': min(self.training_losses) if self.training_losses else float('inf')
        }
