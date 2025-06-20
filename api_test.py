#!/usr/bin/env python3
"""
Test script for the LLM API
"""
import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_model_info():
    """Test model info endpoint"""
    print("Testing model info endpoint...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_system_info():
    """Test system info endpoint"""
    print("Testing system info endpoint...")
    response = requests.get(f"{BASE_URL}/system/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_training():
    """Test model training"""
    print("Testing model training...")
    
    # Sample training text
    training_text = """
    Once upon a time, there was a young programmer who wanted to build her own language model.
    She learned about neural networks, transformers, and deep learning.
    Every day, she practiced coding and studied machine learning papers.
    Eventually, she created a beautiful language model that could generate stories.
    The model learned patterns in text and could create new, creative content.
    She was proud of her achievement and shared it with the world.
    """
    
    training_data = {
        "text": training_text,
        "sequence_length": 30,
        "batch_size": 8,
        "learning_rate": 0.01,
        "num_epochs": 5,
        "embed_dim": 64,
        "num_heads": 2,
        "num_layers": 1
    }
    
    response = requests.post(f"{BASE_URL}/model/train", json=training_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()
    
    # Check training status
    print("Checking training status...")
    for i in range(10):
        response = requests.get(f"{BASE_URL}/model/training/status")
        status = response.json()
        print(f"Epoch {status['epoch']}/{status['total_epochs']}, Status: {status['status']}, Loss: {status['loss']:.4f}")
        
        if status['status'] == 'completed':
            print("Training completed!")
            break
        elif status['status'] == 'error':
            print(f"Training failed: {status.get('error', 'Unknown error')}")
            break
        
        time.sleep(2)
    print()

def test_generation():
    """Test text generation"""
    print("Testing text generation...")
    
    generation_data = {
        "prompt": "Once upon a time",
        "max_length": 100,
        "temperature": 0.8,
        "top_k": 10
    }
    
    response = requests.post(f"{BASE_URL}/model/generate", json=generation_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Generated text: {result['generated_text']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_multiple_generation():
    """Test multiple text generation"""
    print("Testing multiple text generation...")
    
    data = {
        "prompt": "The quick brown",
        "num_generations": 3,
        "max_length": 50,
        "temperature": 1.0,
        "top_k": 10
    }
    
    response = requests.post(f"{BASE_URL}/model/generate/multiple", data=data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Generated {result['count']} texts:")
        for i, gen in enumerate(result['generations']):
            print(f"{i+1}. {gen['text']}")
    else:
        print(f"Error: {response.text}")
    print()

def main():
    """Run all tests"""
    print("=== LLM API Test Suite ===\n")
    
    try:
        test_health()
        test_system_info()
        test_model_info()
        test_training()
        test_generation()
        test_multiple_generation()
        
        print("=== All tests completed ===")
        
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to API server. Make sure it's running on port 8000.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()