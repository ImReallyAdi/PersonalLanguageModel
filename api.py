from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import json
import os
from model import SimpleTransformer
from trainer import ModelTrainer
from data_loader import TextDataLoader
from text_generator import TextGenerator
from utils import save_model, load_model, get_device, count_parameters
import asyncio
import threading
import time

app = FastAPI(
    title="My Own LLM API",
    description="API for training and using character-level language models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model state
current_model = None
current_trainer = None
char_to_idx = None
idx_to_char = None
training_progress = {"status": "idle", "epoch": 0, "loss": 0, "total_epochs": 0}

# Request/Response models
class TrainingRequest(BaseModel):
    text: str
    sequence_length: int = 50
    batch_size: int = 16
    learning_rate: float = 0.003
    num_epochs: int = 20
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2

class GenerationRequest(BaseModel):
    prompt: str = ""
    max_length: int = 200
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None

class ModelInfo(BaseModel):
    vocab_size: Optional[int] = None
    total_parameters: Optional[int] = None
    embed_dim: Optional[int] = None
    num_heads: Optional[int] = None
    num_layers: Optional[int] = None
    sequence_length: Optional[int] = None
    device: str
    is_trained: bool

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "My Own LLM API is running", "status": "healthy"}

@app.get("/model/info")
async def get_model_info():
    """Get current model information"""
    global current_model, char_to_idx
    
    device = str(get_device())
    
    if current_model is None:
        return ModelInfo(device=device, is_trained=False)
    
    return ModelInfo(
        vocab_size=len(char_to_idx) if char_to_idx else None,
        total_parameters=count_parameters(current_model),
        embed_dim=current_model.embed_dim,
        num_heads=current_model.num_heads,
        num_layers=current_model.num_layers,
        sequence_length=current_model.sequence_length,
        device=device,
        is_trained=True
    )

@app.post("/model/train")
async def train_model(request: TrainingRequest):
    """Train a new model with the provided text data"""
    global current_model, current_trainer, char_to_idx, idx_to_char, training_progress
    
    try:
        # Validate input
        if len(request.text) < 100:
            raise HTTPException(status_code=400, detail="Text must be at least 100 characters long")
        
        # Initialize training progress
        training_progress = {
            "status": "preparing",
            "epoch": 0,
            "loss": 0,
            "total_epochs": request.num_epochs
        }
        
        # Prepare data
        data_loader = TextDataLoader(
            request.text, 
            request.sequence_length, 
            request.batch_size
        )
        vocab_size = data_loader.vocab_size
        
        # Store character mappings
        char_to_idx = data_loader.char_to_idx
        idx_to_char = data_loader.idx_to_char
        
        # Initialize model
        device = get_device()
        model = SimpleTransformer(
            vocab_size=vocab_size,
            embed_dim=request.embed_dim,
            num_heads=request.num_heads,
            num_layers=request.num_layers,
            sequence_length=request.sequence_length
        ).to(device)
        
        # Initialize trainer
        trainer = ModelTrainer(model, data_loader, request.learning_rate, str(device))
        
        # Store globally
        current_model = model
        current_trainer = trainer
        
        # Start training in background
        asyncio.create_task(train_model_background(request.num_epochs))
        
        return {
            "message": "Training started",
            "model_info": {
                "vocab_size": vocab_size,
                "parameters": count_parameters(model),
                "embed_dim": request.embed_dim,
                "num_heads": request.num_heads,
                "num_layers": request.num_layers
            }
        }
        
    except Exception as e:
        training_progress["status"] = "error"
        raise HTTPException(status_code=500, detail=str(e))

async def train_model_background(num_epochs: int):
    """Background training function"""
    global training_progress, current_trainer
    
    try:
        training_progress["status"] = "training"
        
        for epoch in range(num_epochs):
            if current_trainer is None:
                break
            epoch_loss = current_trainer.train_epoch()
            
            training_progress.update({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "status": "training"
            })
            
            # Small delay to prevent blocking
            await asyncio.sleep(0.01)
        
        training_progress["status"] = "completed"
        
        # Save the trained model
        if current_model is not None:
            model_config = {
                'vocab_size': current_model.vocab_size,
                'embed_dim': current_model.embed_dim,
                'num_heads': current_model.num_heads,
                'num_layers': current_model.num_layers,
                'sequence_length': current_model.sequence_length
            }
            
            save_model(current_model, char_to_idx, idx_to_char, model_config)
        
    except Exception as e:
        training_progress["status"] = "error"
        training_progress["error"] = str(e)

@app.get("/model/training/status")
async def get_training_status():
    """Get current training status"""
    return training_progress

@app.post("/model/generate")
async def generate_text(request: GenerationRequest):
    """Generate text using the trained model"""
    global current_model, char_to_idx, idx_to_char
    
    if current_model is None:
        raise HTTPException(status_code=400, detail="No trained model available")
    
    if char_to_idx is None or idx_to_char is None:
        raise HTTPException(status_code=400, detail="Model vocabulary not available")
    
    try:
        generator = TextGenerator(
            current_model,
            char_to_idx,
            idx_to_char,
            str(get_device())
        )
        
        generated_text = generator.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        
        return {
            "generated_text": generated_text,
            "prompt": request.prompt,
            "parameters": {
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/generate/multiple")
async def generate_multiple_texts(
    prompt: str = Form(""),
    num_generations: int = Form(3),
    max_length: int = Form(200),
    temperature: float = Form(1.0),
    top_k: Optional[int] = Form(None)
):
    """Generate multiple text samples"""
    global current_model, char_to_idx, idx_to_char
    
    if current_model is None:
        raise HTTPException(status_code=400, detail="No trained model available")
    
    try:
        generator = TextGenerator(
            current_model,
            char_to_idx,
            idx_to_char,
            str(get_device())
        )
        
        generations = generator.generate_multiple(
            prompt=prompt,
            num_generations=num_generations,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k or 10
        )
        
        return {
            "generations": generations,
            "prompt": prompt,
            "count": len(generations)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/upload-training-data")
async def upload_training_data(file: UploadFile = File(...)):
    """Upload a text file for training"""
    if not file.filename or not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")
    
    try:
        content = await file.read()
        text = content.decode('utf-8')
        
        if len(text) < 100:
            raise HTTPException(status_code=400, detail="Text must be at least 100 characters long")
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "text_length": len(text),
            "text_preview": text[:200] + "..." if len(text) > 200 else text
        }
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/load")
async def load_saved_model():
    """Load a previously saved model"""
    global current_model, char_to_idx, idx_to_char
    
    try:
        model_data = load_model()
        
        if model_data is None:
            raise HTTPException(status_code=404, detail="No saved model found")
        
        current_model = model_data['model']
        char_to_idx = model_data['char_to_idx']
        idx_to_char = model_data['idx_to_char']
        
        return {
            "message": "Model loaded successfully",
            "model_info": {
                "vocab_size": len(char_to_idx),
                "parameters": count_parameters(current_model),
                "embed_dim": current_model.embed_dim,
                "num_heads": current_model.num_heads,
                "num_layers": current_model.num_layers
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/vocabulary")
async def get_vocabulary():
    """Get model vocabulary information"""
    global char_to_idx, idx_to_char
    
    if char_to_idx is None:
        raise HTTPException(status_code=400, detail="No model vocabulary available")
    
    # Get character frequency information
    chars = list(char_to_idx.keys())
    
    return {
        "vocab_size": len(chars),
        "characters": chars[:50],  # Show first 50 characters
        "total_characters": len(chars),
        "special_tokens": [char for char in chars if char.startswith('<') and char.endswith('>')]
    }

@app.get("/model/probabilities")
async def get_token_probabilities(context: str, top_k: int = 10):
    """Get top-k token probabilities for a given context"""
    global current_model, char_to_idx, idx_to_char
    
    if current_model is None:
        raise HTTPException(status_code=400, detail="No trained model available")
    
    try:
        generator = TextGenerator(
            current_model,
            char_to_idx,
            idx_to_char,
            str(get_device())
        )
        
        probabilities = generator.get_token_probabilities(context, top_k)
        
        return {
            "context": context,
            "top_probabilities": probabilities
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/model")
async def clear_model():
    """Clear the current model from memory"""
    global current_model, current_trainer, char_to_idx, idx_to_char, training_progress
    
    current_model = None
    current_trainer = None
    char_to_idx = None
    idx_to_char = None
    training_progress = {"status": "idle", "epoch": 0, "loss": 0, "total_epochs": 0}
    
    return {"message": "Model cleared from memory"}

@app.get("/system/info")
async def get_system_info():
    """Get system information"""
    device = get_device()
    
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)