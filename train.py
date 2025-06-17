import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from model import Translator, Config

def train_tokenizer(sentences, vocab_size, special_tokens, lang):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train_from_iterator(sentences, trainer)
    tokenizer.save(f"{lang}_tokenizer.json")
    return tokenizer

class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer_en, tokenizer_fr, max_length):
        self.dataset = dataset
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]['translation']
        source_text = sample['en']
        target_text = sample['fr']
        source_ids = self.tokenizer_en.encode(source_text).ids
        target_ids = [self.tokenizer_fr.token_to_id("<sos>")] + self.tokenizer_fr.encode(target_text).ids + [self.tokenizer_fr.token_to_id("<eos>")]
        if len(source_ids) > self.max_length:
            source_ids = source_ids[:self.max_length]
        if len(target_ids) > self.max_length:
            target_ids = target_ids[:self.max_length]
        return {"source": source_ids, "target": target_ids}

def collate_fn(batch):
    source = [torch.tensor(item["source"], dtype=torch.long) for item in batch]
    target = [torch.tensor(item["target"], dtype=torch.long) for item in batch]
    source_padded = torch.nn.utils.rnn.pad_sequence(source, batch_first=True, padding_value=0)
    target_padded = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=0)
    return {"source": source_padded, "target": target_padded}

def train():
    # Load dataset
    dataset = load_dataset("wmt14", "fr-en", split="train")
    # Optional: Use a subset for testing
    # dataset = dataset.select(range(10000))

    # Extract sentences
    english_sentences = [sample['translation']['en'] for sample in dataset]
    french_sentences = [sample['translation']['fr'] for sample in dataset]

    # Train tokenizers
    vocab_size = 10000
    special_tokens_en = ["<pad>", "<unk>"]
    special_tokens_fr = ["<pad>", "<sos>", "<eos>", "<unk>"]

    if not os.path.exists("en_tokenizer.json"):
        tokenizer_en = train_tokenizer(english_sentences, vocab_size, special_tokens_en, "en")
    else:
        tokenizer_en = Tokenizer.from_file("en_tokenizer.json")

    if not os.path.exists("fr_tokenizer.json"):
        tokenizer_fr = train_tokenizer(french_sentences, vocab_size, special_tokens_fr, "fr")
    else:
        tokenizer_fr = Tokenizer.from_file("fr_tokenizer.json")

    # Get actual vocab sizes and special token IDs
    src_vocab_size = tokenizer_en.get_vocab_size()
    tgt_vocab_size = tokenizer_fr.get_vocab_size()
    SOS = tokenizer_fr.token_to_id("<sos>")
    EOS = tokenizer_fr.token_to_id("<eos>")
    PAD = tokenizer_fr.token_to_id("<pad>")

    # Initialize config with tokenizer-derived values
    config = Config(
        d_model=512,
        n_heads=8,
        N=6,
        context_length=512,
        int_dim=2048,
        SOS=SOS,
        EOS=EOS,
        PAD=PAD
    )
    config.src_vocab_size = src_vocab_size
    config.tgt_vocab_size = tgt_vocab_size

    # Create dataset and dataloader
    max_length = config.context_length
    train_dataset = TranslationDataset(dataset, tokenizer_en, tokenizer_fr, max_length)
    batch_size = 32  # Adjust based on GPU memory
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = Translator(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 10  # Adjust as needed
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad()
            _, loss = model(source, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        # Save model
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()