import torch
import torch.nn as nn
import requests
from torch.utils.data import Dataset, DataLoader
import math


def load_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text.lower()
    text = (
        text.replace(":", " : ")
        .replace(",", " , ")
        .replace(".", " . ")
        .replace(";", " ; ")
        .replace("!", " ! ")
        .replace("?", " ? ")
    )
    words = text.split()
    subset_size = len(words) // 10  
    return " ".join(words[:subset_size])


text = load_shakespeare()
words = text.split()
vocab = sorted(list(set(words)))
vocab.append("<UNK>")
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)


class ShakespeareDataset(Dataset):
    def __init__(self, words, sequence_length=20):
        self.words = words
        self.sequence_length = sequence_length
        self.total_len = len(words) - sequence_length

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        chunk = self.words[idx : idx + self.sequence_length + 1]
        x = torch.tensor(
            [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in chunk[:-1]],
            dtype=torch.long,
        )
        y = torch.tensor(
            [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in chunk[1:]],
            dtype=torch.long,
        )
        return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )

        self.output_layer = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src, tgt=None):
        if tgt is None:
            tgt = src

        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)

        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.output_layer(output)


d_model = 128
nhead = 4
num_layers = 2
sequence_length = 20
batch_size = 64
num_epochs = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de : {device}")

dataset = ShakespeareDataset(words, sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = TransformerLanguageModel(
    vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers
).to(device)

try:
    model.load_state_dict(torch.load("shakespeare_transformer.pth"))
    print("Modèle chargé avec succès!")
except FileNotFoundError:
    print("Aucun modèle existant trouvé. Démarrage d'un nouvel entraînement.")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)
criterion = nn.CrossEntropyLoss()


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 500 == 0:
            print(
                f"    Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}"
            )

    return total_loss / len(dataloader)


def generate_text(model, start_text, max_length=50, temperature=1.2):
    model.eval()
    words = start_text.lower().split()
    device = next(model.parameters()).device

    last_n_words = []
    max_repetitions = 3

    with torch.no_grad():
        for _ in range(max_length):
            input_indices = [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in words]
            input_seq = (
                torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
            )

            output = model(input_seq)
            logits = output[0, -1, :] / temperature

            if last_n_words:
                for word in last_n_words:
                    if word in word_to_idx:
                        logits[word_to_idx[word]] *= 0.7

            probs = torch.softmax(logits, dim=0)

            k = 5
            top_k_probs, top_k_indices = torch.topk(probs, k)
            next_word_idx = top_k_indices[torch.multinomial(top_k_probs, 1)].item()

            next_word = idx_to_word[next_word_idx]

            if len(last_n_words) >= max_repetitions and all(
                w == next_word for w in last_n_words[-max_repetitions:]
            ):
                continue

            words.append(next_word)
            last_n_words.append(next_word)
            if len(last_n_words) > 5:
                last_n_words.pop(0)

            if next_word == "." or next_word == "?" or next_word == "!":
                break

    return " ".join(words)


print(f"Début de l'entraînement avec {num_epochs} époques")
print(f"Taille du vocabulaire: {vocab_size} mots")
print(f"Taille du dataset: {len(dataset)} séquences")
print(f"Nombre de batchs par époque: {len(dataloader)}")
print("=" * 50)

for epoch in range(num_epochs):
    print(f"\nDébut de l'époque {epoch + 1}/{num_epochs}")
    loss = train_epoch(model, dataloader, criterion, optimizer, device)
    scheduler.step(loss)
    print(f"Époque {epoch + 1} terminée, Loss moyenne: {loss:.4f}")

    # Générer un exemple toutes les époques
    print("\nExemple de génération:")
    sample_text = generate_text(model, "First Citizen", max_length=30)
    print(sample_text)

print("\nSauvegarde du modèle...")
torch.save(model.state_dict(), "shakespeare_transformer.pth")
print("Modèle sauvegardé dans 'shakespeare_transformer.pth'")

print("\nGénération d'exemples finaux:")
prompts = [
    "First Citizen",
    "The king commands",
    "In fair Verona",
    "My kingdom for",
    "Friends, Romans, countrymen",
]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    generated = generate_text(model, prompt, max_length=50)
    print(generated)
