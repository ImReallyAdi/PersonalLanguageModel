#!/usr/bin/env python3
"""
Complete system test for the Personal Language Model
Tests API, training, and generation functionality
"""

import requests
import json
import time
import sys

API_BASE = "http://localhost:8000"

def test_api_health():
    """Test API health endpoint"""
    print("Testing API health...")
    try:
        response = requests.get(f"{API_BASE}/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print("✓ API health check passed")
        return True
    except Exception as e:
        print(f"✗ API health check failed: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("Testing model info...")
    try:
        response = requests.get(f"{API_BASE}/model/info")
        assert response.status_code == 200
        data = response.json()
        print(f"✓ Model info: {data['total_parameters']} parameters, vocab_size: {data['vocab_size']}")
        return True
    except Exception as e:
        print(f"✗ Model info failed: {e}")
        return False

def test_training():
    """Test model training"""
    print("Testing model training...")
    try:
        training_data = {
            "text": "Hello I am an AI assistant. I can help with questions and conversations. I enjoy talking about science technology history arts and life.",
            "sequence_length": 20,
            "batch_size": 8,
            "learning_rate": 0.01,
            "num_epochs": 3
        }
        
        response = requests.post(f"{API_BASE}/model/train", json=training_data)
        assert response.status_code == 200
        data = response.json()
        print(f"✓ Training started: {data['message']}")
        return True
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False

def test_chat():
    """Test chat functionality"""
    print("Testing chat...")
    
    # Wait a moment for training to complete
    time.sleep(3)
    
    test_messages = ["hello", "how are you", "tell me about AI"]
    
    for message in test_messages:
        try:
            response = requests.get(f"{API_BASE}/api/chat/{message}")
            assert response.status_code == 200
            data = response.json()
            
            if "error" in data:
                print(f"✗ Chat error for '{message}': {data['error']}")
                return False
            else:
                print(f"✓ Chat '{message}' -> '{data['response'][:50]}...'")
        except Exception as e:
            print(f"✗ Chat failed for '{message}': {e}")
            return False
    
    return True

def test_generation():
    """Test text generation endpoint"""
    print("Testing text generation...")
    try:
        gen_data = {
            "prompt": "The future of AI",
            "max_length": 100,
            "temperature": 0.8
        }
        
        response = requests.post(f"{API_BASE}/model/generate", json=gen_data)
        assert response.status_code == 200
        data = response.json()
        print(f"✓ Generation: '{data['generated_text'][:50]}...'")
        return True
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Personal Language Model System Test ===\n")
    
    tests = [
        test_api_health,
        test_model_info,
        test_training,
        test_chat,
        test_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for deployment.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())