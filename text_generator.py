import torch
import torch.nn.functional as F
import numpy as np

class TextGenerator:
    """Text generation class for the trained language model."""
    
    def __init__(self, model, char_to_idx, idx_to_char, device='cpu'):
        self.model = model
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.device = device
        self.model.eval()
    
    def generate(self, prompt="", max_length=200, temperature=1.0, top_k=None, top_p=None):
        """Generate text given a prompt."""
        # Handle empty prompt
        if not prompt:
            prompt = self.idx_to_char.get(np.random.randint(len(self.idx_to_char)), '')
        
        # Encode the prompt
        input_ids = [self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 0)) for char in prompt]
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        generated_text = prompt
        
        with torch.no_grad():
            for _ in range(max_length - len(prompt)):
                # Ensure input doesn't exceed model's sequence length
                if input_tensor.size(1) > self.model.sequence_length:
                    input_tensor = input_tensor[:, -self.model.sequence_length:]
                
                # Get model predictions
                outputs = self.model(input_tensor)
                
                # Get logits for the last position
                next_token_logits = outputs[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    next_token_logits = self.top_k_filtering(next_token_logits, top_k)
                
                # Apply top-p filtering
                if top_p is not None:
                    next_token_logits = self.top_p_filtering(next_token_logits, top_p)
                
                # Convert to probabilities
                probabilities = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probabilities, num_samples=1)
                
                # Convert to character and add to generated text
                next_char = self.idx_to_char.get(next_token.item(), '')
                generated_text += next_char
                
                # Update input tensor
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
                
                # Stop generation if we hit end-of-sequence token
                if next_char == '<EOS>':
                    break
        
        return generated_text
    
    def top_k_filtering(self, logits, top_k):
        """Apply top-k filtering to logits."""
        top_k = min(top_k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        
        # Create a mask for top-k tokens
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(0, top_k_indices, top_k_logits)
        
        return mask
    
    def top_p_filtering(self, logits, top_p):
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        
        # Set logits to -inf for tokens to be removed
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def generate_with_beam_search(self, prompt="", max_length=200, beam_size=5, temperature=1.0):
        """Generate text using beam search decoding."""
        if not prompt:
            prompt = self.idx_to_char.get(np.random.randint(len(self.idx_to_char)), '')
        
        # Encode the prompt
        input_ids = [self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 0)) for char in prompt]
        
        # Initialize beams
        beams = [(input_ids, 0.0)]  # (sequence, score)
        
        with torch.no_grad():
            for _ in range(max_length - len(prompt)):
                candidates = []
                
                for sequence, score in beams:
                    # Convert to tensor
                    input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
                    
                    # Ensure input doesn't exceed model's sequence length
                    if input_tensor.size(1) > self.model.sequence_length:
                        input_tensor = input_tensor[:, -self.model.sequence_length:]
                    
                    # Get model predictions
                    outputs = self.model(input_tensor)
                    
                    # Get logits for the last position
                    next_token_logits = outputs[0, -1, :] / temperature
                    
                    # Convert to log probabilities
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # Get top beam_size candidates
                    top_log_probs, top_indices = torch.topk(log_probs, beam_size)
                    
                    for i in range(beam_size):
                        new_sequence = sequence + [top_indices[i].item()]
                        new_score = score + top_log_probs[i].item()
                        candidates.append((new_sequence, new_score))
                
                # Select top beam_size candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_size]
                
                # Check for end-of-sequence
                if any(self.idx_to_char.get(seq[-1], '') == '<EOS>' for seq, _ in beams):
                    break
        
        # Get the best sequence
        best_sequence, _ = beams[0]
        
        # Decode to text
        generated_text = ''.join([self.idx_to_char.get(idx, '') for idx in best_sequence])
        
        return generated_text
    
    def generate_multiple(self, prompt="", num_generations=3, max_length=200, temperature=1.0, top_k=10):
        """Generate multiple text samples."""
        generations = []
        
        for i in range(num_generations):
            # Add some randomness by varying temperature slightly
            temp = temperature + np.random.normal(0, 0.1)
            temp = max(0.1, min(2.0, temp))  # Clamp between 0.1 and 2.0
            
            generated = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temp,
                top_k=top_k
            )
            
            generations.append({
                'text': generated,
                'temperature': temp,
                'length': len(generated)
            })
        
        return generations
    
    def calculate_perplexity(self, text):
        """Calculate perplexity of the model on given text."""
        # Encode text
        input_ids = [self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 0)) for char in text]
        
        if len(input_ids) < 2:
            return float('inf')
        
        total_log_prob = 0
        num_tokens = 0
        
        with torch.no_grad():
            for i in range(1, len(input_ids)):
                # Get context
                start_idx = max(0, i - self.model.sequence_length)
                context = input_ids[start_idx:i]
                target = input_ids[i]
                
                # Convert to tensor
                input_tensor = torch.tensor([context], dtype=torch.long).to(self.device)
                
                # Get model predictions
                outputs = self.model(input_tensor)
                
                # Get logits for the last position
                logits = outputs[0, -1, :]
                
                # Convert to log probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Add log probability of target token
                total_log_prob += log_probs[target].item()
                num_tokens += 1
        
        # Calculate perplexity
        avg_log_prob = total_log_prob / num_tokens if num_tokens > 0 else 0
        perplexity = np.exp(-avg_log_prob)
        
        return perplexity
    
    def get_token_probabilities(self, context, top_k=10):
        """Get top-k token probabilities given a context."""
        # Encode context
        input_ids = [self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 0)) for char in context]
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Ensure input doesn't exceed model's sequence length
        if input_tensor.size(1) > self.model.sequence_length:
            input_tensor = input_tensor[:, -self.model.sequence_length:]
        
        with torch.no_grad():
            # Get model predictions
            outputs = self.model(input_tensor)
            
            # Get logits for the last position
            logits = outputs[0, -1, :]
            
            # Convert to probabilities
            probabilities = F.softmax(logits, dim=-1)
            
            # Get top-k probabilities
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.idx_to_char)))
            
            # Convert to readable format
            results = []
            for prob, idx in zip(top_probs, top_indices):
                char = self.idx_to_char.get(idx.item(), '<UNK>')
                results.append({
                    'character': char,
                    'probability': prob.item(),
                    'index': idx.item()
                })
        
        return results
