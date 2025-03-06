from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import math

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.2,
            batch_first=True,
            activation=nn.GELU()
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer(src)
        return self.output_layer(output)

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 50
    temperature: float = 0.7  

def load_model_and_vocab():
    try:
        checkpoint = torch.load("data_transformer.pth", map_location=torch.device('cpu'), weights_only=False)
        vocab_data = checkpoint['vocab_data']
        word_to_idx = vocab_data['word_to_idx']
        idx_to_word = vocab_data['idx_to_word']
        vocab_size = vocab_data['vocab_size']
        
        config = {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 4,
            'vocab_size': vocab_size
        }
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransformerLanguageModel(**config).to(device)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model and vocabulary (size: {vocab_size}) loaded successfully!")
        except RuntimeError as e:
            print("Error loading model weights:", str(e))
            raise
            
        return model, config, word_to_idx, idx_to_word
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please train the model first using train.py")
        raise

model, MODEL_CONFIG, word_to_idx, idx_to_word = load_model_and_vocab()
model.eval()

def generate_text(model, start_text, max_length=50, temperature=0.9):
    if start_text.strip() == "":
        return ""
        
    model.eval()
    # Clean input text
    cleaned_text = start_text.lower().replace('"', '').strip()
    words = [w for w in cleaned_text.split() if w.strip()]
    initial_prompt_words = set(words)
    device = next(model.parameters()).device
    
    # Tracking structures
    word_frequencies = {word: 1 for word in words}
    recent_words = set()  # Track recent words for local repetition
    banned_words = set()  # Words that shouldn't appear again
    
    # Sentence control
    end_tokens = [".", "?", "!"]
    sentence_count = 0
    max_sentences = 2
    min_words_per_sentence = 5
    max_words_per_sentence = 10
    current_sentence_words = len(words)
    
    # Common words that should connect ideas
    connectors = {"and", "but", "because", "while", "though", "however", "when", "if", "as"}
    pronouns = {"he", "she", "they", "it", "we", "you", "i"}
    
    with torch.inference_mode():
        while len(words) < max_length:
            # Use limited context
            context_window = words[-8:] if len(words) > 8 else words
            
            input_indices = [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in context_window]
            input_seq = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)

            output = model(input_seq)
            logits = output[0, -1, :] / temperature
            
            # Sample from top-k
            k = min(50, len(word_to_idx))
            top_k_probs, top_k_indices = torch.topk(logits, k)
            probs = torch.softmax(top_k_probs, dim=0)
            
            valid_words = []
            valid_probs = []
            
            # Filter candidates
            for i, idx in enumerate(top_k_indices):
                word = idx_to_word[idx.item()]
                
                # Skip if:
                if (word in banned_words or  # Globally banned
                    word in recent_words or  # Recently used
                    word in word_frequencies and word_frequencies[word] >= 2 or  # Used twice
                    (len(words) > 0 and word == words[-1]) or  # Immediate repetition
                    (word in initial_prompt_words and len(words) < 8)):  # Too soon to repeat prompt
                    continue
                
                # Boost probability for connectors at appropriate positions
                boost = 1.0
                if current_sentence_words > 3 and word in connectors:
                    boost = 1.2
                if len(words) > 0 and words[-1] in end_tokens and word in pronouns:
                    boost = 1.3
                
                valid_words.append(word)
                valid_probs.append(probs[i].item() * boost)
                
                if len(valid_words) >= 20:  # Limit candidates
                    break
            
            # Handle no valid words
            if not valid_words:
                if current_sentence_words >= min_words_per_sentence:
                    words.append(".")
                    sentence_count += 1
                    if sentence_count >= max_sentences:
                        break
                    current_sentence_words = 0
                    recent_words.clear()
                    continue
                else:
                    # Reset sentence if too short
                    words = words[:-current_sentence_words]
                    current_sentence_words = 0
                    recent_words.clear()
                    continue
            
            # Sample next word
            valid_probs = torch.tensor(valid_probs)
            valid_probs = valid_probs / valid_probs.sum()
            next_word = valid_words[torch.multinomial(valid_probs, 1).item()]
            
            # Update tracking
            word_frequencies[next_word] = word_frequencies.get(next_word, 0) + 1
            recent_words.add(next_word)
            if len(recent_words) > 5:  # Keep track of last 5 words
                recent_words.remove(next(iter(recent_words)))
            
            current_sentence_words += 1
            words.append(next_word)
            
            # Handle sentence endings
            if next_word in end_tokens:
                if current_sentence_words < min_words_per_sentence:
                    words.pop()
                    continue
                sentence_count += 1
                if sentence_count >= max_sentences:
                    break
                current_sentence_words = 0
                recent_words.clear()
            
            # Force end if sentence too long
            if current_sentence_words >= max_words_per_sentence:
                words.append(".")
                sentence_count += 1
                if sentence_count >= max_sentences:
                    break
                current_sentence_words = 0
                recent_words.clear()

    text = " ".join(words)
    if not any(text.strip().endswith(token) for token in end_tokens):
        text += "."
        
    return text

@app.post("/generate")
async def generate(request: GenerateRequest):
    if not request.prompt.strip():
        return {"error": "Empty prompt"}
    
    try:
        # Dynamic temperature based on prompt
        base_temp = 0.85
        prompt_words = len(request.prompt.split())
        temperature = base_temp + (0.05 * min(prompt_words, 3))
        
        generated_text = generate_text(
            model, 
            request.prompt,
            max_length=40,  # Shorter length for more control
            temperature=temperature
        )
        
        return {
            "generated_text": generated_text,
            "model_config": {
                "temperature": temperature,
                "max_length": request.max_length,
                **MODEL_CONFIG
            }
        }
    except Exception as e:
        return {"error": str(e)}