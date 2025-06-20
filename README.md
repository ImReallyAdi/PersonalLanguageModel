# My Own LLM - Educational Language Model System

A complete educational platform for training and using character-level language models, featuring both a web interface and REST API.

## Features

### Web Interface (Streamlit)
- **Interactive Training**: Train models with custom text data
- **Real-time Progress**: Watch training progress with live loss charts
- **Text Generation**: Generate creative text with customizable parameters
- **Model Management**: Save, load, and analyze trained models
- **Vocabulary Analysis**: Explore model vocabulary and statistics

### REST API (FastAPI)
- **Model Training**: `/model/train` - Train new models programmatically
- **Text Generation**: `/model/generate` - Generate text with various parameters
- **Multiple Generation**: `/model/generate/multiple` - Generate multiple text samples
- **Model Info**: `/model/info` - Get detailed model information
- **Training Status**: `/model/training/status` - Monitor training progress
- **File Upload**: `/model/upload-training-data` - Upload training text files
- **Vocabulary**: `/model/vocabulary` - Access model vocabulary
- **Token Probabilities**: `/model/probabilities` - Get token probability distributions

## Quick Start

### Web Interface
1. Access the web interface at: `http://localhost:5000`
2. Choose "Data & Training" to train a new model
3. Select sample data or upload your own text
4. Configure training parameters and start training
5. Use "Text Generation" to create new text
6. View "Model Info" for detailed statistics

### API Usage
1. API is available at: `http://localhost:8000`
2. View interactive documentation at: `http://localhost:8000/docs`

#### Example API Calls

**Train a Model:**
```bash
curl -X POST "http://localhost:8000/model/train" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your training text here...",
    "sequence_length": 50,
    "batch_size": 16,
    "learning_rate": 0.003,
    "num_epochs": 20,
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 2
  }'
```

**Generate Text:**
```bash
curl -X POST "http://localhost:8000/model/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_length": 200,
    "temperature": 1.0,
    "top_k": 10
  }'
```

**Check Training Status:**
```bash
curl "http://localhost:8000/model/training/status"
```

## Model Architecture

The system uses a character-level transformer model with:
- **Multi-head attention**: Captures long-range dependencies
- **Positional encoding**: Maintains sequence order information
- **Layer normalization**: Stabilizes training
- **Dropout regularization**: Prevents overfitting

## Training Parameters

- **Sequence Length**: Length of input sequences (10-200)
- **Batch Size**: Number of sequences per batch (1-64)
- **Learning Rate**: Training step size (0.001-0.1)
- **Epochs**: Number of training iterations (1-100)
- **Embedding Dimension**: Size of character embeddings (64-512)
- **Attention Heads**: Number of attention mechanisms (2-8)
- **Layers**: Number of transformer layers (1-8)

## Generation Parameters

- **Temperature**: Controls randomness (0.1-2.0)
- **Top-K**: Limits vocabulary to top K tokens
- **Top-P**: Nucleus sampling threshold
- **Max Length**: Maximum generated text length

## Technical Details

### Dependencies
- PyTorch: Neural network framework
- Streamlit: Web interface
- FastAPI: REST API framework
- Matplotlib: Visualization
- NumPy: Numerical computations

### Model Components
- `model.py`: Transformer architecture
- `trainer.py`: Training loop and optimization
- `data_loader.py`: Text preprocessing and batching
- `text_generator.py`: Text generation algorithms
- `utils.py`: Utility functions and model persistence

### Interfaces
- `app.py`: Streamlit web application
- `api.py`: FastAPI REST server

## System Requirements

- Python 3.11+
- CUDA (optional, for GPU acceleration)
- 4GB+ RAM recommended
- Text data for training

## Educational Purpose

This system is designed for learning about:
- Language model architecture
- Transformer attention mechanisms
- Character-level text processing
- Neural network training
- Text generation techniques
- API development
- Web interface design

## API Documentation

Complete API documentation is available at `/docs` when the server is running, providing:
- Interactive endpoint testing
- Request/response schemas
- Parameter descriptions
- Example usage

## Troubleshooting

**Training Issues:**
- Ensure text data is at least 100 characters
- Adjust learning rate if loss doesn't decrease
- Reduce batch size if memory errors occur

**Generation Issues:**
- Train a model before generating text
- Check model vocabulary compatibility
- Adjust temperature for different creativity levels

**API Issues:**
- Verify server is running on port 8000
- Check request format matches API schemas
- Monitor training status endpoint for progress