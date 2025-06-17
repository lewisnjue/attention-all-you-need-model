import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Union

@dataclass
class Config:
    d_model: int = 512
    n_heads: int = 8
    N: int = 6
    context_length: int = 1000
    int_dim: int = 2048
    SOS: int = 1  # Start of Sequence token, change based on the tokenizer 
    EOS: int = 2  # End of Sequence token, change based on the tokenizer (corrected from EOF to EOS)

class FFN(nn.Module):
    def __init__(self, d_model: int, int_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(d_model, int_dim)
        self.layer2 = nn.Linear(int_dim, d_model)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.layer2(self.act(self.layer1(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Union[torch.Tensor, None] = None):
        B, T, D = q.shape
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask[None, None, :, :], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out), attn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, int_dim: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim=d_model, num_heads=nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FFN(d_model=d_model, int_dim=int_dim)

    def forward(self, x: torch.Tensor):
        attn_output, _ = self.self_attn(x, x, x, mask=None)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, int_dim: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim=d_model, num_heads=nhead)
        self.enc_dec_attn = MultiHeadAttention(embed_dim=d_model, num_heads=nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = FFN(d_model=d_model, int_dim=int_dim)

    def forward(self, decoder_input: torch.Tensor, encoder_output: torch.Tensor):
        B, T, C = decoder_input.shape
        mask = torch.triu(torch.ones(T, T, device=decoder_input.device), diagonal=1).bool()
        self_attn_output, _ = self.self_attn(decoder_input, decoder_input, decoder_input, mask=mask)
        decoder_input = self.norm1(decoder_input + self_attn_output)
        enc_dec_output, _ = self.enc_dec_attn(decoder_input, encoder_output, encoder_output, mask=None)
        decoder_input = self.norm2(decoder_input + enc_dec_output)
        ff_output = self.feed_forward(decoder_input)
        decoder_input = self.norm3(decoder_input + ff_output)
        return decoder_input

class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config.d_model, config.n_heads, config.int_dim)
            for _ in range(config.N)
        ])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config.d_model, config.n_heads, config.int_dim)
            for _ in range(config.N)
        ])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor):
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x

class Translator(nn.Module):
    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        # Placeholder vocab sizes; adjust based on your dataset
        self.src_vocab_size = 10000
        self.tgt_vocab_size = 10000
        self.encoder_embedding = nn.Embedding(self.src_vocab_size, config.d_model)
        self.decoder_embedding = nn.Embedding(self.tgt_vocab_size, config.d_model)
        self.encoder_positional_encoding = nn.Embedding(config.context_length, config.d_model)
        self.decoder_positional_encoding = nn.Embedding(config.context_length, config.d_model)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.output_projection = nn.Linear(config.d_model, self.tgt_vocab_size)

    def forward(self, idx: torch.Tensor, target: torch.Tensor, encoder_output: Union[torch.Tensor, None] = None):
        B, T_src = idx.shape
        assert T_src <= self.config.context_length, f"Source sequence length must be <= {self.config.context_length}"
        # Encoder
        if encoder_output is None:
            src_embedding = self.encoder_embedding(idx)
            src_pos = self.encoder_positional_encoding(torch.arange(T_src, device=idx.device))
            src_input = src_embedding + src_pos.unsqueeze(0)
            encoder_output = self.encoder(src_input)
        # Decoder
        B, T_tgt = target.shape
        assert T_tgt <= self.config.context_length, f"Target sequence length must be <= {self.config.context_length}"
        decoder_input = target[:, :-1]
        T_dec = decoder_input.size(1)
        dec_embedding = self.decoder_embedding(decoder_input)
        dec_pos = self.decoder_positional_encoding(torch.arange(T_dec, device=target.device))
        dec_input = dec_embedding + dec_pos.unsqueeze(0)
        decoder_output = self.decoder(dec_input, encoder_output)
        logits = self.output_projection(decoder_output)
        # Loss
        labels = target[:, 1:]
        loss = F.cross_entropy(logits.view(-1, self.tgt_vocab_size), labels.view(-1), ignore_index=0)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_length: Union[int, None] = None):
        if max_length is None:
            max_length = self.config.context_length
        self.eval()
        with torch.no_grad():
            B, T = idx.shape
            src_embedding = self.encoder_embedding(idx)
            src_pos = self.encoder_positional_encoding(torch.arange(T, device=idx.device))
            src_input = src_embedding + src_pos.unsqueeze(0)
            encoder_output = self.encoder(src_input)
            # Start with SOS token from config
            generated = torch.full((B, 1), self.config.SOS, dtype=torch.long, device=idx.device)
            for _ in range(max_length - 1):
                dec_embedding = self.decoder_embedding(generated)
                dec_pos = self.decoder_positional_encoding(torch.arange(generated.size(1), device=idx.device))
                dec_input = dec_embedding + dec_pos.unsqueeze(0)
                dec_output = self.decoder(dec_input, encoder_output)
                logits = self.output_projection(dec_output[:, -1:, :])
                next_token = torch.argmax(logits, dim=-1)
                generated = torch.cat([generated, next_token], dim=1)
                # Stop if all sequences have generated EOS token from config
                if (next_token == self.config.EOS).all():
                    break
        self.train()
        return generated