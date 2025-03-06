import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from dataset import load_movie_dialogs
from model import TransformerLanguageModel

text = load_movie_dialogs()
words = text.split()
vocab = sorted(list(set(words)))
vocab.append("<UNK>")
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)

class MovieDialogDataset(Dataset):
    def __init__(self, words, sequence_length=32):
        self.words = words
        self.sequence_length = sequence_length
        self.total_len = len(words) - sequence_length

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        chunk = self.words[idx:idx + self.sequence_length + 1]
        x = torch.tensor([word_to_idx.get(w, word_to_idx["<UNK>"]) for w in chunk[:-1]], dtype=torch.long)
        y = torch.tensor([word_to_idx.get(w, word_to_idx["<UNK>"]) for w in chunk[1:]], dtype=torch.long)
        return x, y

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"    Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

if __name__ == "__main__":
    d_model = 256
    nhead = 8
    num_layers = 4
    batch_size = 64
    num_epochs = 5
    sequence_length = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MovieDialogDataset(words, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    ).to(device)

    try:
        model.load_state_dict(torch.load("data_transformer.pth")['model_state_dict'])
        print("Loaded existing model successfully!")
    except FileNotFoundError:
        print("No existing model found, starting from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting training with {num_epochs} epochs")
    print(f"Vocabulary size: {vocab_size} words")
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Batches per epoch: {len(dataloader)}")
    print("=" * 50)

    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch + 1}/{num_epochs}")
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1} completed, Average Loss: {avg_loss:.4f}")

        print("Saving model and vocabulary...")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'vocab_data': {
                'word_to_idx': word_to_idx,
                'idx_to_word': idx_to_word,
                'vocab_size': vocab_size
            }
        }
        torch.save(checkpoint, "data_transformer.pth")
        print("Model and vocabulary saved!")

    print("Training completed!")