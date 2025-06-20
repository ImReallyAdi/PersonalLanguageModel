from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import torch
import json
import os
import pickle
import io
import time
from model import SimpleTransformer
from trainer import ModelTrainer
from data_loader import TextDataLoader
from text_generator import TextGenerator
from utils import save_model, load_model, get_device, count_parameters
from database import (
    get_db, init_database, save_model_to_db, load_model_from_db, list_models,
    create_training_session, update_training_session, log_training_epoch,
    log_generation_request, save_training_data, list_training_data, get_training_data,
    get_training_statistics, get_generation_statistics, get_db_session
)
import asyncio
import threading
from datetime import datetime

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
current_model_id = None
current_training_session_id = None
training_progress = {"status": "idle", "epoch": 0, "loss": 0, "total_epochs": 0}

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_database()
    
    # Try to load the most recent model
    try:
        db = get_db_session()
        models = list_models(db, limit=1)
        if models:
            model_data = load_model_from_db(db, models[0]['id'])
            if model_data:
                global current_model, char_to_idx, idx_to_char, current_model_id
                
                # Load model
                current_model = SimpleTransformer(
                    vocab_size=model_data['vocab_size'],
                    embed_dim=model_data['embed_dim'],
                    num_heads=model_data['num_heads'],
                    num_layers=model_data['num_layers'],
                    sequence_length=model_data['sequence_length']
                )
                current_model.load_state_dict(model_data['model_state'])
                current_model.eval()
                
                char_to_idx = model_data['char_to_idx']
                idx_to_char = model_data['idx_to_char']
                current_model_id = models[0]['id']
                
                print(f"✓ Loaded model: {models[0]['name']} ({model_data['total_parameters']:,} parameters)")
        db.close()
    except Exception as e:
        print(f"No saved model found: {e}")
        
    # If no model exists, train a quick demo model
    if current_model is None:
        print("Training demo model...")
        demo_text = """Hello! I am an AI assistant. I can help with questions and conversations. I enjoy talking about science, technology, history, arts, and life. I aim to provide helpful responses. What would you like to talk about? I can help with explanations, writing, problem solving, and conversation. Thank you for chatting! How can I help you today? I hope you are having a great day. Feel free to ask me anything. I love learning and sharing knowledge."""
        
        try:
            # Quick training configuration  
            data_loader = TextDataLoader(demo_text, sequence_length=20, batch_size=8)
            model = SimpleTransformer(
                vocab_size=len(data_loader.char_to_idx),
                embed_dim=64,
                num_heads=2,
                num_layers=2,
                sequence_length=20
            )
            
            trainer = ModelTrainer(model, data_loader, learning_rate=0.01)
            
            # Train for 5 epochs for better quality
            for epoch in range(5):
                loss = trainer.train_epoch()
                print(f"Epoch {epoch+1}/5, Loss: {loss:.4f}")
            
            # Set global variables
            current_model = model
            char_to_idx = data_loader.char_to_idx
            idx_to_char = data_loader.idx_to_char
            current_model_id = "demo"
            
            print(f"✓ Demo model ready! Vocab size: {len(char_to_idx)}")
            print(f"Available characters: {sorted(list(char_to_idx.keys()))}")
            
        except Exception as e:
            print(f"Failed to train demo model: {e}")

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
async def train_model(request: TrainingRequest, db: Session = Depends(get_db)):
    """Train a new model with the provided text data"""
    global current_model, current_trainer, char_to_idx, idx_to_char, training_progress, current_training_session_id
    
    try:
        # Validate input
        if len(request.text) < 100:
            raise HTTPException(status_code=400, detail="Text must be at least 100 characters long")
        
        # Save training data to database
        training_data_id = save_training_data(
            db, 
            f"Training_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
            request.text, 
            "API training request"
        )
        
        # Create training session record
        session_config = {
            'training_text_length': len(request.text),
            'sequence_length': request.sequence_length,
            'batch_size': request.batch_size,
            'learning_rate': request.learning_rate,
            'num_epochs': request.num_epochs,
            'embed_dim': request.embed_dim,
            'num_heads': request.num_heads,
            'num_layers': request.num_layers
        }
        current_training_session_id = create_training_session(db, session_config)
        
        # Initialize training progress
        training_progress = {
            "status": "preparing",
            "epoch": 0,
            "loss": 0,
            "total_epochs": request.num_epochs,
            "session_id": current_training_session_id
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
            "session_id": current_training_session_id,
            "training_data_id": training_data_id,
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
        if current_training_session_id:
            update_training_session(db, current_training_session_id, {
                "status": "error",
                "error_message": str(e),
                "completed_at": datetime.utcnow()
            })
        raise HTTPException(status_code=500, detail=str(e))

async def train_model_background(num_epochs: int):
    """Background training function"""
    global training_progress, current_trainer, current_model, char_to_idx, idx_to_char, current_training_session_id, current_model_id
    
    try:
        training_progress["status"] = "training"
        start_time = datetime.utcnow()
        
        # Get database session
        db = get_db_session()
        
        try:
            epoch_loss = 0.0
            for epoch in range(num_epochs):
                if current_trainer is None:
                    break
                    
                epoch_loss = current_trainer.train_epoch()
                
                # Log epoch to database
                if current_training_session_id:
                    log_training_epoch(
                        db, 
                        current_training_session_id, 
                        epoch + 1, 
                        epoch_loss, 
                        current_trainer.get_learning_rate()
                    )
                
                training_progress.update({
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "status": "training"
                })
                
                # Small delay to prevent blocking
                await asyncio.sleep(0.01)
            
            training_progress["status"] = "completed"
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Save the trained model to database
            if current_model is not None and char_to_idx is not None and idx_to_char is not None:
                # Serialize model weights
                model_buffer = io.BytesIO()
                torch.save(current_model.state_dict(), model_buffer)
                model_buffer.seek(0)
                
                model_data = {
                    'name': f"Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'description': f"Trained via API with {num_epochs} epochs",
                    'vocab_size': current_model.vocab_size,
                    'embed_dim': current_model.embed_dim,
                    'num_heads': current_model.num_heads,
                    'num_layers': current_model.num_layers,
                    'sequence_length': current_model.sequence_length,
                    'total_parameters': count_parameters(current_model),
                    'model_weights': model_buffer.getvalue(),
                    'char_to_idx': char_to_idx,
                    'idx_to_char': idx_to_char
                }
                
                current_model_id = save_model_to_db(db, model_data)
                
                # Update training session with completion
                if current_training_session_id:
                    update_training_session(db, current_training_session_id, {
                        "model_id": current_model_id,
                        "final_loss": epoch_loss,
                        "status": "completed",
                        "completed_at": end_time,
                        "duration_seconds": int(duration)
                    })
        
        finally:
            db.close()
        
    except Exception as e:
        training_progress["status"] = "error"
        training_progress["error"] = str(e)
        
        # Update training session with error
        if current_training_session_id:
            db = get_db_session()
            try:
                update_training_session(db, current_training_session_id, {
                    "status": "error",
                    "error_message": str(e),
                    "completed_at": datetime.utcnow()
                })
            finally:
                db.close()

@app.get("/model/training/status")
async def get_training_status():
    """Get current training status"""
    return training_progress

@app.post("/model/generate")
async def generate_text(request: GenerationRequest, db: Session = Depends(get_db)):
    """Generate text using the trained model"""
    global current_model, char_to_idx, idx_to_char, current_model_id
    
    if current_model is None:
        raise HTTPException(status_code=400, detail="No trained model available")
    
    if char_to_idx is None or idx_to_char is None:
        raise HTTPException(status_code=400, detail="Model vocabulary not available")
    
    try:
        start_time = time.time()
        
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
        
        generation_time = int((time.time() - start_time) * 1000)
        
        # Log generation to database
        if current_model_id:
            log_generation_request(db, {
                'model_id': current_model_id,
                'prompt': request.prompt,
                'generated_text': generated_text,
                'max_length': request.max_length,
                'temperature': request.temperature,
                'top_k': request.top_k,
                'top_p': request.top_p,
                'generation_time_ms': generation_time
            })
        
        return {
            "generated_text": generated_text,
            "prompt": request.prompt,
            "generation_time_ms": generation_time,
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

@app.get("/models/list")
async def list_saved_models(db: Session = Depends(get_db)):
    """List all saved models"""
    models = list_models(db)
    return {"models": models}

@app.post("/model/load")
async def load_specific_model(request: dict, db: Session = Depends(get_db)):
    """Load a specific model by ID"""
    global current_model, char_to_idx, idx_to_char, current_model_id
    
    model_id = request.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    
    try:
        model_data = load_model_from_db(db, model_id)
        if not model_data:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Reconstruct model
        device = get_device()
        model = SimpleTransformer(
            vocab_size=model_data['vocab_size'],
            embed_dim=model_data['embed_dim'],
            num_heads=model_data['num_heads'],
            num_layers=model_data['num_layers'],
            sequence_length=model_data['sequence_length']
        ).to(device)
        
        # Load weights from binary data
        model_weights = torch.load(io.BytesIO(model_data['model_weights']), map_location=device)
        model.load_state_dict(model_weights)
        model.eval()
        
        # Update global state
        current_model = model
        char_to_idx = model_data['char_to_idx']
        idx_to_char = model_data['idx_to_char']
        current_model_id = model_id
        
        return {
            "message": "Model loaded successfully",
            "model_info": {
                "id": model_id,
                "name": model_data['name'],
                "vocab_size": model_data['vocab_size'],
                "parameters": model_data['total_parameters']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/personalities")
async def get_chat_personalities():
    """Get available chat personalities"""
    personalities = {
        "helpful": {
            "name": "Helpful Assistant",
            "description": "A friendly and helpful AI assistant",
            "prompt_template": "You are a helpful AI assistant. User says: {message}\nAssistant:"
        },
        "creative": {
            "name": "Creative Companion", 
            "description": "An imaginative and creative AI",
            "prompt_template": "You are a creative and imaginative AI. User says: {message}\nLet me think creatively about this:"
        },
        "professional": {
            "name": "Professional Consultant",
            "description": "A formal and professional AI consultant", 
            "prompt_template": "You are a professional AI consultant. User says: {message}\nMy professional response:"
        },
        "friendly": {
            "name": "Casual Friend",
            "description": "A friendly and casual AI companion",
            "prompt_template": "You are a friendly and casual AI companion. User says: {message}\nHey there! "
        },
        "wise": {
            "name": "Wise Mentor",
            "description": "A thoughtful and wise AI mentor",
            "prompt_template": "You are a wise and thoughtful AI mentor. User says: {message}\nWith wisdom and reflection:"
        }
    }
    return {"personalities": personalities}

@app.post("/chat/message")
async def chat_message(message: str, personality: str = "helpful", temperature: float = 0.8, max_length: int = 150, db: Session = Depends(get_db)):
    """Send a chat message and get AI response"""
    global current_model, char_to_idx, idx_to_char, current_model_id
    
    if current_model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")
    
    try:
        # Simplified personality templates
        personality_prompts = {
            'helpful': f"User: {message}\nAssistant:",
            'creative': f"User: {message}\nCreative AI:",
            'professional': f"User: {message}\nProfessional response:",
            'friendly': f"User: {message}\nFriend:",
            'wise': f"User: {message}\nWise mentor:"
        }
        
        prompt = personality_prompts.get(personality, personality_prompts['helpful'])
        
        # Generate response
        start_time = time.time()
        generator = TextGenerator(current_model, char_to_idx, idx_to_char, str(get_device()))
        
        generated_text = generator.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=10
        )
        
        generation_time = int((time.time() - start_time) * 1000)
        
        # Extract AI response
        if "Assistant:" in generated_text:
            ai_response = generated_text.split("Assistant:")[-1].strip()
        elif ":" in generated_text:
            ai_response = generated_text.split(":")[-1].strip()
        else:
            ai_response = generated_text[len(prompt):].strip()
        
        return {
            "response": ai_response,
            "time_ms": generation_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/{message}")
async def quick_chat(message: str):
    """Quick chat endpoint for simple GET requests"""
    global current_model, char_to_idx, idx_to_char
    
    if current_model is None:
        return {"error": "No model loaded"}
    
    if char_to_idx is None or idx_to_char is None:
        return {"error": "Character mappings not available"}
    
    try:
        prompt = f"User: {message}\nAI:"
        
        # Ensure prompt contains only characters from our vocabulary
        valid_chars = set(char_to_idx.keys())
        filtered_prompt = ''.join(c if c in valid_chars else ' ' for c in prompt)
        
        generator = TextGenerator(current_model, char_to_idx, idx_to_char, str(get_device()))
        generated_text = generator.generate(
            prompt=filtered_prompt,
            max_length=100,
            temperature=0.8,
            top_k=10
        )
        
        # Extract response
        if "AI:" in generated_text:
            response = generated_text.split("AI:")[-1].strip()
        else:
            response = generated_text[len(filtered_prompt):].strip()
        
        return {"response": response}
        
    except Exception as e:
        return {"error": str(e)}

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