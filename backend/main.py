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
        checkpoint = torch.load("got_transformer.pth", map_location=torch.device('cpu'))
        
        vocab_data = checkpoint['vocab_data']
        word_to_idx = vocab_data['word_to_idx']
        idx_to_word = vocab_data['idx_to_word']
        vocab_size = vocab_data['vocab_size']
        
        config = {
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,
            'vocab_size': vocab_size
        }
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransformerLanguageModel(**config).to(device)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model and vocabulary (size: {vocab_size}) loaded successfully!")
        except RuntimeError as e:
            print("Error loading model weights. Training new model...")
            raise
            
        return model, config, word_to_idx, idx_to_word
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please train the model first using train.py")
        raise

model, MODEL_CONFIG, word_to_idx, idx_to_word = load_model_and_vocab()
model.eval()

def generate_text(model, start_text, max_length=50, temperature=0.7):
    if start_text.strip() == "":
        return ""
        
    model.eval()
    input_text = start_text.lower()
    
    if "death" in input_text or "died" in input_text:
        name = input_text.replace("death", "").replace("died", "").strip()
        words = [name, "met", "their", "fate"]
    else:
        words = input_text.split()
    
    device = next(model.parameters()).device
    
    end_tokens = [".", "?", "!"]
    sentence_count = 0
    max_sentences = 3 
    
    with torch.inference_mode():
        for _ in range(max_length):
            context_window = words[-64:] if len(words) > 64 else words
            
            input_indices = [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in context_window]
            input_seq = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)

            output = model(input_seq)
            logits = output[0, -1, :] / temperature

            k = min(20, int(len(word_to_idx) * 0.2))
            top_k_probs, top_k_indices = torch.topk(logits, k)
            probs = torch.softmax(top_k_probs, dim=0)
            
            # Éviter les répétitions
            if len(words) > 2:
                last_word = words[-1]
                last_2_words = words[-2:]
                
                # Pénaliser les mots récemment utilisés
                for i, idx in enumerate(top_k_indices):
                    word = idx_to_word[idx.item()]
                    if word == last_word:
                        probs[i] *= 0.05  # Pénalité plus forte pour les répétitions
                    elif word in last_2_words:
                        probs[i] *= 0.2
            
            next_word_idx = top_k_indices[torch.multinomial(probs, 1)].item()
            next_word = idx_to_word[next_word_idx]
            
            # Éviter les répétitions de phrases
            if next_word in end_tokens:
                sentence_count += 1
                if sentence_count >= max_sentences:
                    words.append(next_word)
                    break
            
            words.append(next_word)

    text = " ".join(words)
    
    if not any(text.strip().endswith(token) for token in end_tokens):
        text += "."
        
    return text

@app.post("/generate")
async def generate(request: GenerateRequest):
    if not request.prompt.strip():
        return {"error": "Empty prompt"}
    
    try:
        temperature = 0.8  
        
        generated_text = generate_text(
            model, 
            request.prompt,
            max_length=request.max_length,
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