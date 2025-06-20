import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, embed_dim, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:embed_dim//2])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SimpleTransformer(nn.Module):
    """Simple character-level transformer model for educational purposes."""
    
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, 
                 sequence_length=50, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, sequence_length)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for transformer."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src, src_mask=None):
        """Forward pass through the model."""
        # src shape: (batch_size, sequence_length)
        
        # Token embedding
        src = self.token_embedding(src) * math.sqrt(self.embed_dim)
        
        # Add positional encoding
        src = src.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        src = self.positional_encoding(src)
        src = src.transpose(0, 1)  # (batch_size, seq_len, embed_dim)
        
        # Apply dropout
        src = self.dropout(src)
        
        # Generate causal mask if not provided
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        # Transformer forward pass
        output = self.transformer(src, src_mask)
        
        # Project to vocabulary size
        output = self.output_projection(output)
        
        return output
    
    def generate_next_token(self, input_ids, temperature=1.0, top_k=None):
        """Generate next token given input sequence."""
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            logits = self.forward(input_ids)
            
            # Get logits for the last position
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                
                # Create a mask for top-k tokens
                mask = torch.full_like(next_token_logits, float('-inf'))
                mask.scatter_(1, top_k_indices, top_k_logits)
                next_token_logits = mask
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
        return next_token

class SimpleRNN(nn.Module):
    """Alternative simple RNN model for comparison."""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.1):
        super(SimpleRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        """Forward pass through the RNN model."""
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        
        # Output projection
        output = self.output_layer(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state for LSTM."""
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (hidden, cell)
