"""
Vercel serverless function entry point for the FastAPI application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
import time
import os

# Global variables for model state
current_model = None
char_to_idx = None
idx_to_char = None
current_model_id = None
training_progress = {"status": "idle", "epoch": 0, "loss": 0, "total_epochs": 0}

# Simple in-memory storage for Vercel (no PostgreSQL)
models_storage = {}
training_sessions = {}

app = FastAPI(title="Personal Language Model API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simplified model classes for Vercel
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=2, num_layers=2, sequence_length=20):
        super().__init__()
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(sequence_length, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, src):
        seq_len = src.size(1)
        embeddings = self.embedding(src)
        embeddings += self.pos_encoding[:seq_len].unsqueeze(0)
        
        mask = torch.triu(torch.ones(seq_len, seq_len)) == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        output = self.transformer(embeddings, mask=mask)
        output = self.output_layer(output)
        return output

class TextDataLoader:
    def __init__(self, text, sequence_length=20, batch_size=8):
        self.text = text
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        
        # Build vocabulary
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Encode text
        self.encoded_text = [self.char_to_idx[ch] for ch in text]

class TextGenerator:
    def __init__(self, model, char_to_idx, idx_to_char, device='cpu'):
        self.model = model
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.device = device
    
    def generate(self, prompt="", max_length=100, temperature=1.0, top_k=10):
        self.model.eval()
        with torch.no_grad():
            # Encode prompt
            if prompt:
                context = [self.char_to_idx.get(ch, 0) for ch in prompt]
            else:
                context = [0]
            
            # Generate
            for _ in range(max_length):
                if len(context) > self.model.sequence_length:
                    context = context[-self.model.sequence_length:]
                
                input_tensor = torch.tensor([context], dtype=torch.long)
                outputs = self.model(input_tensor)
                logits = outputs[0, -1, :] / temperature
                
                # Top-k sampling
                if top_k:
                    values, indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(0, indices, values)
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                context.append(int(next_token))
            
            # Decode
            generated_text = ''.join([self.idx_to_char.get(idx, '') for idx in context])
            return generated_text

@app.on_event("startup")
async def startup_event():
    global current_model, char_to_idx, idx_to_char, current_model_id
    
    # Train a small demo model for immediate use
    demo_text = "Hello I am an AI assistant. I can help with questions and conversations. I enjoy talking about science technology history arts and life. I aim to provide helpful responses. What would you like to talk about? I can help with explanations writing problem solving and conversation. Thank you for chatting! How can I help you today? I hope you are having a great day. Feel free to ask me anything. I love learning and sharing knowledge."
    
    try:
        data_loader = TextDataLoader(demo_text, sequence_length=15, batch_size=4)
        model = SimpleTransformer(
            vocab_size=len(data_loader.char_to_idx),
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            sequence_length=15
        )
        
        # Quick training (2 epochs for fast startup)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        
        for epoch in range(2):
            total_loss = 0
            for i in range(0, len(data_loader.encoded_text) - data_loader.sequence_length, data_loader.sequence_length):
                if i + data_loader.sequence_length + 1 >= len(data_loader.encoded_text):
                    break
                
                inputs = torch.tensor([data_loader.encoded_text[i:i+data_loader.sequence_length]], dtype=torch.long)
                targets = torch.tensor([data_loader.encoded_text[i+1:i+data_loader.sequence_length+1]], dtype=torch.long)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        current_model = model
        char_to_idx = data_loader.char_to_idx
        idx_to_char = data_loader.idx_to_char
        current_model_id = "demo"
        
        print(f"Demo model ready! Vocab size: {len(char_to_idx)}")
        
    except Exception as e:
        print(f"Failed to train demo model: {e}")

@app.get("/")
async def root():
    return {"message": "Personal Language Model API", "status": "running", "timestamp": datetime.utcnow()}

@app.get("/api/chat/{message}")
async def quick_chat(message: str):
    global current_model, char_to_idx, idx_to_char
    
    if current_model is None:
        return {"error": "No model loaded"}
    
    if char_to_idx is None or idx_to_char is None:
        return {"error": "Character mappings not available"}
    
    try:
        prompt = f"User: {message}\nAI:"
        
        # Filter prompt to only include known characters
        valid_chars = set(char_to_idx.keys())
        filtered_prompt = ''.join(c if c in valid_chars else ' ' for c in prompt)
        
        generator = TextGenerator(current_model, char_to_idx, idx_to_char)
        generated_text = generator.generate(
            prompt=filtered_prompt,
            max_length=60,
            temperature=0.8,
            top_k=8
        )
        
        # Extract response
        if "AI:" in generated_text:
            response = generated_text.split("AI:")[-1].strip()
        else:
            response = generated_text[len(filtered_prompt):].strip()
        
        return {"response": response}
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/model/info")
async def get_model_info():
    global current_model, char_to_idx
    
    if current_model is None:
        return {
            "vocab_size": None,
            "total_parameters": 0,
            "embed_dim": None,
            "num_heads": None,
            "num_layers": None,
            "sequence_length": None,
            "device": "cpu",
            "is_trained": False
        }
    
    total_params = sum(p.numel() for p in current_model.parameters())
    
    return {
        "vocab_size": len(char_to_idx) if char_to_idx else None,
        "total_parameters": total_params,
        "embed_dim": current_model.embed_dim,
        "num_heads": 2,
        "num_layers": 1,
        "sequence_length": current_model.sequence_length,
        "device": "cpu",
        "is_trained": True
    }

@app.get("/docs")
async def get_docs():
    return {
        "title": "Personal Language Model API",
        "description": "A lightweight LLM API for Vercel deployment",
        "endpoints": {
            "/": "API status",
            "/api/chat/{message}": "Chat with the AI model",
            "/model/info": "Get model information",
            "/docs": "This documentation"
        },
        "usage": "Visit /api/chat/hello to test the chat functionality"
    }

# Export the app for Vercel
handler = app