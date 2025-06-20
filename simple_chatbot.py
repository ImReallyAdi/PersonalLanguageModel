import streamlit as st
import requests
import time

# Configure the page
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

def check_model_status():
    """Check if a model is loaded"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json().get('is_trained', False)
    except:
        pass
    return False

def send_message(message):
    """Send message to API and get response"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/chat/{message}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('response', 'No response')
        else:
            return "Error: Could not get response"
    except Exception as e:
        return f"Error: {str(e)}"

def quick_train_model():
    """Train a simple model quickly"""
    sample_text = """Hello! I am an AI assistant. I can help answer questions and have conversations.
    I enjoy talking about many topics including science, technology, arts, and everyday life.
    Feel free to ask me anything you'd like to know or discuss.
    I aim to be helpful and provide useful information.
    What would you like to talk about today?
    I can help with explanations, creative writing, problem solving, and general conversation.
    Thank you for chatting with me!""" * 5
    
    training_config = {
        "text": sample_text,
        "sequence_length": 25,
        "batch_size": 4,
        "learning_rate": 0.01,
        "num_epochs": 2,
        "embed_dim": 32,
        "num_heads": 2,
        "num_layers": 1
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/model/train", json=training_config, timeout=10)
        return response.status_code == 200
    except:
        return False

def main():
    st.title("🤖 AI Chatbot")
    
    # Check model status
    if not st.session_state.model_loaded:
        st.session_state.model_loaded = check_model_status()
    
    if not st.session_state.model_loaded:
        st.warning("No model loaded. Training a quick demo model...")
        
        if st.button("Train Demo Model", type="primary"):
            with st.spinner("Training model (this will take 1-2 minutes)..."):
                if quick_train_model():
                    # Wait for training to complete
                    for _ in range(30):  # Wait up to 30 seconds
                        try:
                            status_response = requests.get(f"{API_BASE_URL}/model/training/status", timeout=5)
                            if status_response.status_code == 200:
                                status = status_response.json()
                                if status['status'] == 'completed':
                                    st.session_state.model_loaded = True
                                    st.success("Model trained successfully!")
                                    st.rerun()
                                elif status['status'] == 'error':
                                    st.error("Training failed")
                                    break
                        except:
                            pass
                        time.sleep(2)
                else:
                    st.error("Failed to start training")
        return
    
    # Chat interface
    st.success("Model ready! Start chatting below.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_message(prompt)
            st.write(response)
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick buttons
    if len(st.session_state.messages) == 0:
        st.subheader("Try these:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👋 Say hello"):
                st.session_state.messages.append({"role": "user", "content": "Hello!"})
                st.rerun()
        
        with col2:
            if st.button("❓ Ask a question"):
                st.session_state.messages.append({"role": "user", "content": "What can you help me with?"})
                st.rerun()
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()