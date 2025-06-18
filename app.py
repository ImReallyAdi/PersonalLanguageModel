import streamlit as st
import torch
import torch.nn as nn
import os
import json
import matplotlib.pyplot as plt
from model import SimpleTransformer
from trainer import ModelTrainer
from data_loader import TextDataLoader
from text_generator import TextGenerator
from utils import save_model, load_model, get_device

# Page configuration
st.set_page_config(
    page_title="My Own LLM - Educational LLM Training System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'char_to_idx' not in st.session_state:
    st.session_state.char_to_idx = None
if 'idx_to_char' not in st.session_state:
    st.session_state.idx_to_char = None

def main():
    st.title("🤖 My Own LLM")
    st.markdown("### Educational LLM Training and Inference System")
    st.markdown("Train your own character-level language model and generate text!")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", 
                               ["Data & Training", "Text Generation", "Model Info"])
    
    if page == "Data & Training":
        data_and_training_page()
    elif page == "Text Generation":
        text_generation_page()
    elif page == "Model Info":
        model_info_page()

def data_and_training_page():
    st.header("📚 Data Preparation & Model Training")
    
    # Data input section
    st.subheader("1. Training Data")
    
    # Option to use sample data or upload custom data
    data_option = st.radio("Choose data source:", 
                          ["Use sample data", "Upload custom text file", "Enter custom text"])
    
    text_data = ""
    
    if data_option == "Use sample data":
        if os.path.exists("sample_data.txt"):
            with open("sample_data.txt", "r", encoding="utf-8") as f:
                text_data = f.read()
            st.success(f"Sample data loaded! ({len(text_data)} characters)")
            with st.expander("Preview sample data"):
                st.text(text_data[:500] + "..." if len(text_data) > 500 else text_data)
        else:
            st.error("Sample data file not found!")
    
    elif data_option == "Upload custom text file":
        uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
        if uploaded_file is not None:
            text_data = str(uploaded_file.read(), "utf-8")
            st.success(f"File uploaded! ({len(text_data)} characters)")
            with st.expander("Preview uploaded data"):
                st.text(text_data[:500] + "..." if len(text_data) > 500 else text_data)
    
    elif data_option == "Enter custom text":
        text_data = st.text_area("Enter your training text:", 
                                height=200, 
                                placeholder="Enter text to train your model on...")
        if text_data:
            st.success(f"Text entered! ({len(text_data)} characters)")
    
    # Training configuration
    st.subheader("2. Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sequence_length = st.slider("Sequence Length", 10, 200, 50, 
                                   help="Length of input sequences for training")
        batch_size = st.slider("Batch Size", 1, 64, 16,
                              help="Number of sequences processed together")
        learning_rate = st.select_slider("Learning Rate", 
                                        options=[0.001, 0.003, 0.01, 0.03, 0.1],
                                        value=0.003,
                                        help="How fast the model learns")
    
    with col2:
        num_epochs = st.slider("Number of Epochs", 1, 100, 20,
                              help="How many times to go through the entire dataset")
        embed_dim = st.slider("Embedding Dimension", 64, 512, 128,
                             help="Size of character embeddings")
        num_heads = st.selectbox("Number of Attention Heads", [2, 4, 8], index=1,
                                help="Number of attention heads in transformer")
        num_layers = st.slider("Number of Layers", 1, 8, 2,
                              help="Number of transformer layers")
    
    # Training section
    st.subheader("3. Model Training")
    
    if text_data and len(text_data) > 100:
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Preparing data and initializing model..."):
                # Prepare data
                data_loader = TextDataLoader(text_data, sequence_length, batch_size)
                vocab_size = data_loader.vocab_size
                
                # Store character mappings in session state
                st.session_state.char_to_idx = data_loader.char_to_idx
                st.session_state.idx_to_char = data_loader.idx_to_char
                
                # Initialize model
                device = get_device()
                model = SimpleTransformer(
                    vocab_size=vocab_size,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    sequence_length=sequence_length
                ).to(device)
                
                # Initialize trainer
                trainer = ModelTrainer(model, data_loader, learning_rate, device)
                
                # Store in session state
                st.session_state.model = model
                st.session_state.trainer = trainer
                
                st.success("Model initialized! Starting training...")
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            loss_chart = st.empty()
            
            training_losses = []
            
            # Training loop
            for epoch in range(num_epochs):
                epoch_loss = st.session_state.trainer.train_epoch()
                training_losses.append(epoch_loss)
                
                # Update progress
                progress = (epoch + 1) / num_epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")
                
                # Update loss chart
                if len(training_losses) > 1:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(training_losses)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title("Training Loss")
                    ax.grid(True)
                    loss_chart.pyplot(fig)
                    plt.close()
            
            # Save training history
            st.session_state.training_history = training_losses
            st.session_state.is_trained = True
            
            st.success("🎉 Training completed!")
            st.balloons()
            
            # Save model
            model_path = save_model(
                st.session_state.model, 
                st.session_state.char_to_idx, 
                st.session_state.idx_to_char,
                {
                    'vocab_size': vocab_size,
                    'embed_dim': embed_dim,
                    'num_heads': num_heads,
                    'num_layers': num_layers,
                    'sequence_length': sequence_length
                }
            )
            st.info(f"Model saved to: {model_path}")
    
    else:
        st.warning("Please provide training data with at least 100 characters to start training.")
    
    # Display training history if available
    if st.session_state.training_history:
        st.subheader("4. Training Progress")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(st.session_state.training_history)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Over Time")
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

def text_generation_page():
    st.header("✨ Text Generation")
    
    # Check if model is trained or load existing model
    if not st.session_state.is_trained:
        st.subheader("Load Existing Model")
        if st.button("Load Saved Model"):
            try:
                model_data = load_model()
                if model_data:
                    st.session_state.model = model_data['model']
                    st.session_state.char_to_idx = model_data['char_to_idx']
                    st.session_state.idx_to_char = model_data['idx_to_char']
                    st.session_state.is_trained = True
                    st.success("Model loaded successfully!")
                    st.rerun()
                else:
                    st.error("No saved model found. Please train a model first.")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    
    if st.session_state.is_trained and st.session_state.model:
        st.subheader("Generate Text")
        
        # Generation parameters
        col1, col2 = st.columns(2)
        
        with col1:
            prompt = st.text_input("Enter a prompt:", 
                                 value="The quick brown",
                                 help="Starting text for generation")
            max_length = st.slider("Maximum length to generate", 50, 1000, 200)
        
        with col2:
            temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1,
                                  help="Higher values make output more random")
            top_k = st.slider("Top-K sampling", 1, 50, 10,
                            help="Consider only top K most likely characters")
        
        if st.button("🎭 Generate Text", type="primary"):
            if prompt:
                with st.spinner("Generating text..."):
                    try:
                        generator = TextGenerator(
                            st.session_state.model,
                            st.session_state.char_to_idx,
                            st.session_state.idx_to_char,
                            get_device()
                        )
                        
                        generated_text = generator.generate(
                            prompt=prompt,
                            max_length=max_length,
                            temperature=temperature,
                            top_k=top_k
                        )
                        
                        st.subheader("Generated Text:")
                        st.text_area("", value=generated_text, height=300, disabled=True)
                        
                        # Download option
                        st.download_button(
                            label="📥 Download Generated Text",
                            data=generated_text,
                            file_name="generated_text.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating text: {str(e)}")
            else:
                st.warning("Please enter a prompt to generate text.")
    
    else:
        st.info("No trained model available. Please train a model first or load an existing one.")

def model_info_page():
    st.header("📊 Model Information")
    
    if st.session_state.model and st.session_state.is_trained:
        # Model architecture info
        st.subheader("Model Architecture")
        
        # Count parameters
        total_params = sum(p.numel() for p in st.session_state.model.parameters())
        trainable_params = sum(p.numel() for p in st.session_state.model.parameters() if p.requires_grad)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Parameters", f"{total_params:,}")
        
        with col2:
            st.metric("Trainable Parameters", f"{trainable_params:,}")
        
        with col3:
            if st.session_state.char_to_idx:
                st.metric("Vocabulary Size", len(st.session_state.char_to_idx))
        
        # Model details
        st.subheader("Model Details")
        
        if hasattr(st.session_state.model, 'embed_dim'):
            st.write(f"**Embedding Dimension:** {st.session_state.model.embed_dim}")
        if hasattr(st.session_state.model, 'num_heads'):
            st.write(f"**Number of Attention Heads:** {st.session_state.model.num_heads}")
        if hasattr(st.session_state.model, 'num_layers'):
            st.write(f"**Number of Layers:** {st.session_state.model.num_layers}")
        
        # Vocabulary preview
        if st.session_state.char_to_idx:
            st.subheader("Vocabulary")
            chars = list(st.session_state.char_to_idx.keys())
            st.write(f"**Characters in vocabulary:** {len(chars)}")
            
            # Show character distribution
            char_display = []
            for i, char in enumerate(chars[:50]):  # Show first 50 characters
                if char == ' ':
                    char_display.append('[SPACE]')
                elif char == '\n':
                    char_display.append('[NEWLINE]')
                elif char == '\t':
                    char_display.append('[TAB]')
                else:
                    char_display.append(char)
            
            st.write("**Sample characters:**", " | ".join(char_display))
            if len(chars) > 50:
                st.write(f"... and {len(chars) - 50} more characters")
        
        # Training history
        if st.session_state.training_history:
            st.subheader("Training Statistics")
            
            final_loss = st.session_state.training_history[-1]
            min_loss = min(st.session_state.training_history)
            epochs_trained = len(st.session_state.training_history)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Final Loss", f"{final_loss:.4f}")
            
            with col2:
                st.metric("Best Loss", f"{min_loss:.4f}")
            
            with col3:
                st.metric("Epochs Trained", epochs_trained)
        
        # Device info
        st.subheader("System Information")
        device = get_device()
        st.write(f"**Training Device:** {device}")
        st.write(f"**PyTorch Version:** {torch.__version__}")
        
        if torch.cuda.is_available():
            st.write(f"**CUDA Available:** Yes")
            st.write(f"**CUDA Device:** {torch.cuda.get_device_name(0)}")
        else:
            st.write(f"**CUDA Available:** No (using CPU)")
    
    else:
        st.info("No model information available. Please train a model first.")

if __name__ == "__main__":
    main()
