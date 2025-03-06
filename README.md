# Movie Dialog Generator 🎬

A sophisticated text generation web application powered by a custom transformer model, trained on movie dialogues. The system generates contextually relevant responses while maintaining natural conversation flow.

## 🌟 Features

- Custom transformer-based language model
- Real-time text generation
- Text-to-speech functionality with British English accent
- Temperature-controlled text generation
- Responsive web interface
- Advanced repetition and coherence control

## 🏗️ Architecture

### Model Architecture

The core of the application is a custom TransformerEncoder model with the following specifications:
- Embedding dimension: 256
- 8 attention heads
- 4 transformer layers
- GELU activation
- Dropout rate: 0.2
- AdamW optimizer with weight decay

```python
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
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
```

### Text Generation Features

- Temperature-based sampling
- Dynamic repetition penalty
- Intelligent sentence structure control
- Logical connector boosting
- Context-aware word selection

## 🛠️ Tech Stack

### Backend
- Python 3.12+
- FastAPI
- PyTorch
- KaggleHub (for dataset)

### Frontend
- Next.js 15 (App Router)
- TypeScript
- TailwindCSS
- Web Speech API

## 📦 Installation

### Prerequisites
- Python 3.12+
- Node.js 20+
- CUDA-capable GPU (recommended)
- Git

### Step 1: Clone the Repository
```bash
git clone git@github.com:N0rooo/ai-project.git
cd ai-project
```

### Step 2: Backend Setup
```bash
# Create and activate virtual environment
cd backend
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


### Step 3: Frontend Setup
```bash
cd new-app
npm install
```

## 🚀 Running the Application

### 1. Start the Backend Server
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 2. Start the Frontend Development Server
```bash
cd new-app
npm run dev
```

The application will be available at `http://localhost:3000`

## 🎯 Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Enter a prompt in the text input field
3. Click "Generate Text" to create new dialogue
4. Use the "Speak Text" button to hear the generated text

## 🔧 Configuration

### Model Parameters
You can adjust the model parameters in `backend/train.py`:

```python
# Training parameters
d_model = 256
nhead = 8
num_layers = 4
batch_size = 64
num_epochs = 5
sequence_length = 32

# Optimizer settings
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=0.01
)
```

### Generation Parameters
Adjust text generation settings in `backend/main.py`:

```python
@app.post("/generate")
async def generate(request: GenerationRequest):
    return {
        "generated_text": generate_text(
            model,
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
    }
```

## 🎓 Training

To train the model on your own data:
```bash
cd backend
python train.py
```

The model will be saved as `data_transformer.pth`
