import streamlit as st
import requests
import json
import time
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="Model Training",
    page_icon="🎓",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def train_model_api(config):
    """Send training request to API"""
    try:
        response = requests.post(f"{API_BASE_URL}/model/train", json=config)
        return response.status_code, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return 500, str(e)

def get_training_status():
    """Get current training status"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/training/status")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"status": "idle", "epoch": 0, "loss": 0, "total_epochs": 0}

def main():
    st.title("🎓 Model Training")
    st.markdown("Train your own language model with custom data")
    
    # Training configuration
    st.header("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Settings")
        text_option = st.radio(
            "Training Data Source:",
            ["Enter Text", "Upload File", "Use Sample Data"]
        )
        
        training_text = ""
        
        if text_option == "Enter Text":
            training_text = st.text_area(
                "Enter your training text:",
                height=200,
                placeholder="Enter text to train your model on..."
            )
        elif text_option == "Upload File":
            uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
            if uploaded_file is not None:
                training_text = str(uploaded_file.read(), "utf-8")
                st.success(f"File loaded! ({len(training_text)} characters)")
        else:
            # Use sample data
            sample_text = """Once upon a time, in a land far away, there lived a young programmer named Alice. She was passionate about artificial intelligence and dreamed of creating her own language model.

Alice spent countless hours reading about neural networks, transformers, and natural language processing. She learned about embeddings, attention mechanisms, and the mathematics behind deep learning.

One day, Alice decided to build her first language model. She started with a simple character-level approach, understanding that each character would be predicted based on the previous sequence of characters."""
            training_text = sample_text
            st.info("Using sample training data")
            with st.expander("Preview sample data"):
                st.text(training_text)
    
    with col2:
        st.subheader("Model Parameters")
        sequence_length = st.slider("Sequence Length", 10, 200, 50)
        batch_size = st.slider("Batch Size", 1, 32, 8)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.001, 0.003, 0.01, 0.03, 0.1],
            value=0.003
        )
        num_epochs = st.slider("Number of Epochs", 1, 50, 10)
        
        st.subheader("Architecture")
        embed_dim = st.slider("Embedding Dimension", 32, 256, 64)
        num_heads = st.selectbox("Attention Heads", [2, 4, 8], index=0)
        num_layers = st.slider("Number of Layers", 1, 4, 1)
    
    # Training section
    st.header("Start Training")
    
    if training_text and len(training_text) > 100:
        config = {
            "text": training_text,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers
        }
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                with st.spinner("Initializing training..."):
                    status_code, response = train_model_api(config)
                    
                    if status_code == 200:
                        st.success("Training started successfully!")
                        st.json(response)
                        st.rerun()
                    else:
                        st.error(f"Training failed: {response}")
        
        # Show training progress if training is active
        status = get_training_status()
        if status["status"] in ["preparing", "training"]:
            st.header("Training Progress")
            
            progress = status["epoch"] / status["total_epochs"] if status["total_epochs"] > 0 else 0
            st.progress(progress)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Epoch", f"{status['epoch']}/{status['total_epochs']}")
            with col2:
                st.metric("Status", status["status"].title())
            with col3:
                if status["loss"] > 0:
                    st.metric("Loss", f"{status['loss']:.4f}")
            
            # Auto-refresh every 2 seconds during training
            if status["status"] == "training":
                time.sleep(2)
                st.rerun()
        
        elif status["status"] == "completed":
            st.success("🎉 Training completed successfully!")
            if st.button("Go to Chat"):
                st.switch_page("chatbot_app.py")
        
        elif status["status"] == "error":
            st.error(f"Training failed: {status.get('error', 'Unknown error')}")
    
    else:
        st.warning("Please provide training data with at least 100 characters.")
    
    # Training history and tips
    st.header("Training Tips")
    
    with st.expander("How to get better results"):
        st.markdown("""
        **Data Quality:**
        - Use diverse, well-written text
        - Include examples of the style you want to generate
        - More data generally leads to better results
        
        **Model Size:**
        - Larger models (more layers/dimensions) can capture more complexity
        - But they also take longer to train and need more data
        
        **Training Parameters:**
        - Lower learning rates are safer but slower
        - More epochs can improve quality but watch for overfitting
        - Batch size affects training speed and memory usage
        """)

if __name__ == "__main__":
    main()