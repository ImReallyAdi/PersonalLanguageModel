#!/usr/bin/env python3
"""
Simple command-line chatbot demo using the API
"""
import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def load_sample_model():
    """Load a sample model for demonstration"""
    print("Training a quick demo model...")
    
    sample_text = """Hello! I am an AI assistant. I can help you with various tasks.
    I can answer questions, provide information, help with creative writing, and have conversations.
    I aim to be helpful, harmless, and honest in all my interactions.
    Feel free to ask me anything you'd like to know or discuss.
    I enjoy learning about different topics and helping people solve problems.
    What would you like to talk about today?"""
    
    training_config = {
        "text": sample_text * 10,  # Repeat for more training data
        "sequence_length": 30,
        "batch_size": 4,
        "learning_rate": 0.01,
        "num_epochs": 3,
        "embed_dim": 64,
        "num_heads": 2,
        "num_layers": 1
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/model/train", json=training_config)
        if response.status_code == 200:
            print("Training started...")
            
            # Wait for training to complete
            while True:
                status_response = requests.get(f"{API_BASE_URL}/model/training/status")
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"Training progress: {status['epoch']}/{status['total_epochs']} - Status: {status['status']}")
                    
                    if status['status'] == 'completed':
                        print("Training completed!")
                        return True
                    elif status['status'] == 'error':
                        print(f"Training failed: {status.get('error', 'Unknown error')}")
                        return False
                
                time.sleep(2)
        else:
            print(f"Failed to start training: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def chat_with_model():
    """Interactive chat session"""
    print("\n🤖 AI Chatbot Demo")
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 40)
    
    personalities = ["helpful", "creative", "professional", "friendly", "wise"]
    current_personality = "helpful"
    
    while True:
        try:
            user_input = input(f"\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("- quit: Exit the chat")
                print("- personality [name]: Change AI personality")
                print(f"- Available personalities: {', '.join(personalities)}")
                continue
            elif user_input.lower().startswith('personality '):
                new_personality = user_input.split(' ', 1)[1].lower()
                if new_personality in personalities:
                    current_personality = new_personality
                    print(f"Switched to {new_personality} personality")
                else:
                    print(f"Unknown personality. Available: {', '.join(personalities)}")
                continue
            elif not user_input:
                continue
            
            # Send message to API
            response = requests.post(f"{API_BASE_URL}/chat/message", params={
                "message": user_input,
                "personality": current_personality,
                "temperature": 0.8,
                "max_length": 100
            })
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['ai_response']
                print(f"\n🤖 AI ({current_personality}): {ai_response}")
                print(f"    [Generated in {result['generation_time_ms']}ms]")
            else:
                print(f"Error: {response.text}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main demo function"""
    print("🚀 Starting AI Chatbot Demo")
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code != 200:
            print("API server is not running. Please start it first.")
            return
    except:
        print("Cannot connect to API server. Please start it first.")
        return
    
    # Check if model is available
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        if response.status_code == 200:
            model_info = response.json()
            if not model_info['is_trained']:
                print("No trained model found. Training a demo model...")
                if not load_sample_model():
                    print("Failed to train model. Exiting.")
                    return
            else:
                print("Using existing trained model")
        else:
            print("Cannot check model status")
            return
    except Exception as e:
        print(f"Error checking model: {e}")
        return
    
    # Start chat session
    chat_with_model()

if __name__ == "__main__":
    main()