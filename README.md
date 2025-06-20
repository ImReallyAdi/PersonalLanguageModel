# Custom LLM Training & Chat System

A complete implementation of a language model training and inference system built with PyTorch, featuring multiple interfaces and deployment options.

## 🚀 Live Demo

**GitHub Pages Demo:** [View Live Demo](https://yourusername.github.io/your-repo-name)

## 📋 Features

- **Custom Transformer Model**: Character-level transformer architecture
- **Training Interface**: Web-based model training with real-time progress
- **Multiple Chat Interfaces**: Streamlit, HTML, and API endpoints
- **Database Integration**: PostgreSQL for storing models and conversations
- **REST API**: Complete FastAPI backend with training and inference endpoints
- **GitHub Pages Ready**: Deployable frontend for easy sharing

## 🏗️ Architecture

```
├── Training Interface (Streamlit) - Port 5000
├── Simple Chatbot (Streamlit) - Port 5002  
├── HTML Chat Interface - Port 5003
├── API Server (FastAPI) - Port 8000
└── PostgreSQL Database
```

## 🚀 Quick Start (Local Development)

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. **Install dependencies**
```bash
pip install torch matplotlib streamlit fastapi uvicorn sqlalchemy psycopg2-binary
```

3. **Start the services**
```bash
# Start API server
python api.py

# Start training interface (new terminal)
streamlit run app.py --server.port 5000

# Start simple chat (new terminal)  
streamlit run simple_chatbot.py --server.port 5002

# Start HTML interface (new terminal)
python chat_server.py
```

4. **Access the interfaces**
- Training: http://localhost:5000
- Simple Chat: http://localhost:5002
- HTML Chat: http://localhost:5003/quick_chat.html
- API Docs: http://localhost:8000/docs

## 🌐 GitHub Pages Deployment

### Step 1: Deploy API to Replit

1. Fork this repository to your GitHub account
2. Import to Replit from GitHub
3. Run the project - Replit will automatically start the API server
4. Note your Replit app URL (e.g., `https://your-app.yourusername.repl.co`)

### Step 2: Enable GitHub Pages

1. Go to your GitHub repository settings
2. Navigate to "Pages" section
3. Set source to "Deploy from a branch"
4. Select "main" branch and "/ (root)" folder
5. Save settings

### Step 3: Configure API URL

1. Visit your GitHub Pages URL (https://yourusername.github.io/your-repo-name)
2. In the API URL field, enter your Replit app URL
3. Click "Test Connection" to verify
4. Start chatting!

## 📡 API Endpoints

### Training
- `POST /model/train` - Train a new model
- `GET /model/training/status` - Check training progress
- `GET /model/info` - Get model information

### Inference  
- `GET /api/chat/{message}` - Quick chat endpoint
- `POST /chat/message` - Advanced chat with parameters
- `POST /generate` - Text generation with custom settings

### Models
- `GET /models/list` - List saved models
- `POST /model/load` - Load a specific model
- `GET /model/vocabulary` - Get model vocabulary

## 🔧 Configuration

### Model Parameters
```python
{
    "sequence_length": 20,    # Input sequence length
    "batch_size": 8,         # Training batch size
    "learning_rate": 0.01,   # Learning rate
    "num_epochs": 3,         # Training epochs
    "embed_dim": 64,         # Embedding dimension
    "num_heads": 2,          # Attention heads
    "num_layers": 2          # Transformer layers
}
```

### Environment Variables
```bash
DATABASE_URL=postgresql://user:pass@host:port/db
PGHOST=localhost
PGPORT=5432
PGUSER=your_user
PGPASSWORD=your_password
PGDATABASE=your_db
```

## 📊 Database Schema

The system uses PostgreSQL with the following tables:
- `models` - Stored trained models
- `training_sessions` - Training history and metrics
- `training_epochs` - Individual epoch results
- `generation_requests` - Text generation logs
- `training_data` - Training datasets

## 🎯 Usage Examples

### Training a Model
```python
import requests

response = requests.post('http://localhost:8000/model/train', json={
    "text": "Your training text here...",
    "sequence_length": 20,
    "batch_size": 8,
    "num_epochs": 5
})
```

### Generating Text
```python
# Quick chat
response = requests.get('http://localhost:8000/api/chat/hello')

# Advanced generation
response = requests.post('http://localhost:8000/generate', json={
    "prompt": "Hello, I am",
    "max_length": 100,
    "temperature": 0.8
})
```

## 🔒 Security Notes

- Never commit API keys or sensitive credentials
- Use environment variables for configuration
- Enable CORS appropriately for your deployment
- Consider rate limiting for production use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🐛 Troubleshooting

### Common Issues

**Connection Error**: Ensure your API server is running and accessible
**Training Fails**: Check that embed_dim is divisible by num_heads
**Memory Issues**: Reduce batch_size or sequence_length for large models
**CORS Errors**: Add your domain to the CORS settings in api.py

### Support

For issues and questions:
1. Check the troubleshooting section above
2. Review API documentation at `/docs`
3. Open an issue on GitHub with detailed information

---

Built with PyTorch, FastAPI, and Streamlit