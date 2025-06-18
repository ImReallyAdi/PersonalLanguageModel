import torch
import os
import json
import pickle
from datetime import datetime

def get_device():
    """Get the best available device (CUDA if available, otherwise CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def save_model(model, char_to_idx, idx_to_char, model_config, save_dir='saved_models'):
    """Save trained model and associated data."""
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save paths
    model_path = os.path.join(save_dir, f'model_{timestamp}.pth')
    vocab_path = os.path.join(save_dir, f'vocab_{timestamp}.pkl')
    config_path = os.path.join(save_dir, f'config_{timestamp}.json')
    
    # Save model state dict
    torch.save(model.state_dict(), model_path)
    
    # Save vocabulary mappings
    vocab_data = {
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char
    }
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    
    # Save model configuration
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Create a latest symlink-like reference file
    latest_path = os.path.join(save_dir, 'latest_model.txt')
    with open(latest_path, 'w') as f:
        f.write(f"{model_path}\n{vocab_path}\n{config_path}")
    
    return model_path

def load_model(save_dir='saved_models'):
    """Load the latest saved model and associated data."""
    latest_path = os.path.join(save_dir, 'latest_model.txt')
    
    if not os.path.exists(latest_path):
        return None
    
    try:
        # Read latest model paths
        with open(latest_path, 'r') as f:
            lines = f.read().strip().split('\n')
            model_path, vocab_path, config_path = lines
        
        # Load model configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        # Import model class
        from model import SimpleTransformer
        
        # Initialize model
        model = SimpleTransformer(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            sequence_length=config['sequence_length']
        )
        
        # Load model weights
        device = get_device()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return {
            'model': model,
            'char_to_idx': vocab_data['char_to_idx'],
            'idx_to_char': vocab_data['idx_to_char'],
            'config': config
        }
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_number(num):
    """Format large numbers with appropriate suffixes."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)

def calculate_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb

def create_training_summary(model, training_history, vocab_size):
    """Create a summary of training results."""
    summary = {
        'model_info': {
            'parameters': count_parameters(model),
            'size_mb': calculate_model_size(model),
            'vocab_size': vocab_size
        },
        'training_info': {
            'epochs': len(training_history),
            'final_loss': training_history[-1] if training_history else None,
            'best_loss': min(training_history) if training_history else None,
            'worst_loss': max(training_history) if training_history else None
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return summary

def clean_generated_text(text):
    """Clean and format generated text."""
    # Remove special tokens
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    for token in special_tokens:
        text = text.replace(token, '')
    
    # Basic text cleaning
    text = text.strip()
    
    # Remove excessive whitespace
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text

def validate_hyperparameters(config):
    """Validate hyperparameter configuration."""
    errors = []
    
    # Check required parameters
    required_params = ['vocab_size', 'embed_dim', 'num_heads', 'num_layers', 'sequence_length']
    for param in required_params:
        if param not in config:
            errors.append(f"Missing required parameter: {param}")
    
    # Check parameter ranges
    if 'embed_dim' in config:
        if config['embed_dim'] < 32 or config['embed_dim'] > 1024:
            errors.append("embed_dim should be between 32 and 1024")
        
        if 'num_heads' in config and config['embed_dim'] % config['num_heads'] != 0:
            errors.append("embed_dim must be divisible by num_heads")
    
    if 'num_heads' in config:
        if config['num_heads'] < 1 or config['num_heads'] > 16:
            errors.append("num_heads should be between 1 and 16")
    
    if 'num_layers' in config:
        if config['num_layers'] < 1 or config['num_layers'] > 12:
            errors.append("num_layers should be between 1 and 12")
    
    if 'sequence_length' in config:
        if config['sequence_length'] < 10 or config['sequence_length'] > 1000:
            errors.append("sequence_length should be between 10 and 1000")
    
    return errors

def get_system_info():
    """Get system information for debugging."""
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': str(get_device())
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    
    return info
