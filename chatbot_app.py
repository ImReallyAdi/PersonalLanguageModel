import streamlit as st
import requests
import json
import time
from datetime import datetime
import os

# Configure the page
st.set_page_config(
    page_title="AI Chatbot - Your Personal LLM",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'chat_settings' not in st.session_state:
    st.session_state.chat_settings = {
        'temperature': 0.8,
        'max_length': 150,
        'top_k': 10,
        'personality': 'helpful'
    }

def get_available_models():
    """Get list of available models from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/list")
        if response.status_code == 200:
            return response.json().get('models', [])
    except:
        pass
    return []

def load_model(model_id):
    """Load a specific model"""
    try:
        response = requests.post(f"{API_BASE_URL}/model/load", json={"model_id": model_id})
        return response.status_code == 200
    except:
        return False

def generate_response(message, settings):
    """Generate response from the chatbot"""
    try:
        # Add personality context to the prompt
        personality_prompts = {
            'helpful': f"You are a helpful AI assistant. User says: {message}\nAssistant:",
            'creative': f"You are a creative and imaginative AI. User says: {message}\nLet me think creatively about this:",
            'professional': f"You are a professional AI consultant. User says: {message}\nMy professional response:",
            'friendly': f"You are a friendly and casual AI companion. User says: {message}\nHey there! ",
            'wise': f"You are a wise and thoughtful AI mentor. User says: {message}\nWith wisdom and reflection:",
        }
        
        prompt = personality_prompts.get(settings['personality'], f"User: {message}\nAI:")
        
        response = requests.post(f"{API_BASE_URL}/model/generate", json={
            "prompt": prompt,
            "max_length": settings['max_length'],
            "temperature": settings['temperature'],
            "top_k": settings['top_k']
        })
        
        if response.status_code == 200:
            generated_text = response.json()['generated_text']
            # Extract only the AI's response part
            if "Assistant:" in generated_text:
                ai_response = generated_text.split("Assistant:")[-1].strip()
            elif "AI:" in generated_text:
                ai_response = generated_text.split("AI:")[-1].strip()
            else:
                # Take everything after the original prompt
                ai_response = generated_text[len(prompt):].strip()
            
            return ai_response
        else:
            return "I'm having trouble generating a response right now. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

def sidebar_controls():
    """Render sidebar controls"""
    st.sidebar.title("🤖 Chatbot Settings")
    
    # Model selection
    st.sidebar.subheader("Model")
    models = get_available_models()
    
    if models:
        model_options = {f"{m['name']} ({m['id'][:8]}...)": m['id'] for m in models}
        selected_model = st.sidebar.selectbox(
            "Choose Model",
            options=list(model_options.keys()),
            key="model_selector"
        )
        
        if selected_model:
            model_id = model_options[selected_model]
            if st.session_state.current_model != model_id:
                if st.sidebar.button("Load Model"):
                    with st.spinner("Loading model..."):
                        if load_model(model_id):
                            st.session_state.current_model = model_id
                            st.sidebar.success("Model loaded!")
                            st.rerun()
                        else:
                            st.sidebar.error("Failed to load model")
    else:
        st.sidebar.warning("No trained models available. Train a model first!")
        if st.sidebar.button("Go to Training"):
            st.sidebar.info("Please use the original training interface at port 5000")
    
    # Chat settings
    st.sidebar.subheader("Chat Settings")
    
    st.session_state.chat_settings['personality'] = st.sidebar.selectbox(
        "Personality",
        options=['helpful', 'creative', 'professional', 'friendly', 'wise'],
        index=0
    )
    
    st.session_state.chat_settings['temperature'] = st.sidebar.slider(
        "Creativity",
        min_value=0.1,
        max_value=2.0,
        value=st.session_state.chat_settings['temperature'],
        step=0.1,
        help="Higher values make responses more creative but less predictable"
    )
    
    st.session_state.chat_settings['max_length'] = st.sidebar.slider(
        "Response Length",
        min_value=50,
        max_value=500,
        value=st.session_state.chat_settings['max_length'],
        step=25
    )
    
    st.session_state.chat_settings['top_k'] = st.sidebar.slider(
        "Vocabulary Focus",
        min_value=5,
        max_value=50,
        value=st.session_state.chat_settings['top_k'],
        step=5,
        help="Lower values make responses more focused"
    )
    
    # Chat management
    st.sidebar.subheader("Chat Management")
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.sidebar.button("Export Chat"):
        chat_data = {
            'timestamp': datetime.now().isoformat(),
            'messages': st.session_state.messages,
            'settings': st.session_state.chat_settings
        }
        st.sidebar.download_button(
            label="Download Chat JSON",
            data=json.dumps(chat_data, indent=2),
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    """Main chatbot interface"""
    st.title("🤖 AI Chatbot")
    st.markdown("Chat with your trained language model!")
    
    # Sidebar controls
    sidebar_controls()
    
    # Check if model is loaded
    if not st.session_state.current_model:
        st.warning("Please select and load a model from the sidebar to start chatting.")
        return
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "timestamp" in message:
                    st.caption(f"Generated at {message['timestamp']}")
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, st.session_state.chat_settings)
            
            st.write(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
    
    # Quick action buttons
    st.subheader("Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎭 Tell me a story"):
            story_prompt = "Tell me an interesting short story"
            st.session_state.messages.append({"role": "user", "content": story_prompt, "timestamp": datetime.now().strftime("%H:%M:%S")})
            st.rerun()
    
    with col2:
        if st.button("💡 Give me ideas"):
            ideas_prompt = "Give me 3 creative ideas for a weekend project"
            st.session_state.messages.append({"role": "user", "content": ideas_prompt, "timestamp": datetime.now().strftime("%H:%M:%S")})
            st.rerun()
    
    with col3:
        if st.button("🧠 Explain something"):
            explain_prompt = "Explain a complex topic in simple terms"
            st.session_state.messages.append({"role": "user", "content": explain_prompt, "timestamp": datetime.now().strftime("%H:%M:%S")})
            st.rerun()
    
    with col4:
        if st.button("🎯 Help me decide"):
            decision_prompt = "Help me make a decision by asking me questions"
            st.session_state.messages.append({"role": "user", "content": decision_prompt, "timestamp": datetime.now().strftime("%H:%M:%S")})
            st.rerun()
    
    # Display chat statistics
    if st.session_state.messages:
        st.subheader("Chat Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Messages", len(st.session_state.messages))
        
        with col2:
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.metric("Your Messages", user_messages)
        
        with col3:
            ai_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            st.metric("AI Responses", ai_messages)

if __name__ == "__main__":
    main()