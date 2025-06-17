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
    context_length: int = 512  # Aligned with train.py max_length
    int_dim: int = 2048
    SOS: int = None  # To be set by tokenizer
    EOS: int = None  # To be set by tokenizer
    PAD: int = 0     # Assuming <pad> is 0, verify with tokenizer

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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Union[torch.Tensor, None] = None, key_padding_mask: Union[torch.Tensor, None] = None):
        B, T_q, D = q.shape
        _, T_k, _ = k.shape
        _, T_v, _ = v.shape
        assert T_k == T_v, "Sequence lengths of k and v must be equal"

        q = self.q_proj(q)  # (B, T_q, D)
        k = self.k_proj(k)  # (B, T_k, D)
        v = self.v_proj(v)  # (B, T_v, D)

        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T_q, head_dim)
        k = k.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T_k, head_dim)
        v = v.view(B, T_v, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T_v, head_dim)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, T_q, T_k)

        if key_padding_mask is not None:
            # key_padding_mask: (B, T_k), True for padding positions
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

        if mask is not None:
            # mask: (T_q, T_k), for causal masking
            scores = scores.masked_fill(mask[None, None, :, :], float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, num_heads, T_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)
        return self.out_proj(out), attn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, int_dim: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim=d_model, num_heads=nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FFN(d_model=d_model, int_dim=int_dim)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor):
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
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

    def forward(self, decoder_input: torch.Tensor, encoder_output: torch.Tensor, tgt_mask: torch.Tensor, src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor):
        self_attn_output, _ = self.self_attn(decoder_input, decoder_input, decoder_input, mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        decoder_input = self.norm1(decoder_input + self_attn_output)
        enc_dec_output, _ = self.enc_dec_attn(decoder_input, encoder_output, encoder_output, key_padding_mask=src_key_padding_mask)
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

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config.d_model, config.n_heads, config.int_dim)
            for _ in range(config.N)
        ])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, tgt_mask: torch.Tensor, src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
        return x

class Translator(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder_embedding = nn.Embedding(config.src_vocab_size, config.d_model)
        self.decoder_embedding = nn.Embedding(config.tgt_vocab_size, config.d_model)
        self.encoder_positional_encoding = nn.Embedding(config.context_length, config.d_model)
        self.decoder_positional_encoding = nn.Embedding(config.context_length, config.d_model)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.output_projection = nn.Linear(config.d_model, config.tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        device = next(self.parameters()).device
        src = src.to(device)
        tgt = tgt.to(device)

        B, T_src = src.shape
        B, T_tgt = tgt.shape

        # Padding masks: True for padding positions
        src_key_padding_mask = (src == self.config.PAD)
        tgt_key_padding_mask = (tgt == self.config.PAD)

        # Encoder
        src_embedding = self.encoder_embedding(src)
        src_pos = self.encoder_positional_encoding(torch.arange(T_src, device=device))
        src_input = src_embedding + src_pos.unsqueeze(0)
        encoder_output = self.encoder(src_input, src_key_padding_mask)

        # Decoder
        decoder_input = tgt[:, :-1]
        T_dec = decoder_input.size(1)
        dec_embedding = self.decoder_embedding(decoder_input)
        dec_pos = self.decoder_positional_encoding(torch.arange(T_dec, device=device))
        dec_input = dec_embedding + dec_pos.unsqueeze(0)

        # Causal mask for decoder self-attention
        tgt_mask = torch.triu(torch.ones(T_dec, T_dec, device=device), diagonal=1).bool()

        decoder_output = self.decoder(dec_input, encoder_output, tgt_mask, src_key_padding_mask, tgt_key_padding_mask[:, :-1])
        logits = self.output_projection(decoder_output)

        # Loss
        labels = tgt[:, 1:]
        loss = F.cross_entropy(logits.view(-1, self.config.tgt_vocab_size), labels.view(-1), ignore_index=self.config.PAD)
        return logits, loss

    def generate(self, src: torch.Tensor, max_length: Union[int, None] = None):
        if max_length is None:
            max_length = self.config.context_length
        self.eval()
        device = next(self.parameters()).device
        src = src.to(device)

        B, T_src = src.shape
        src_key_padding_mask = (src == self.config.PAD)

        # Encoder
        src_embedding = self.encoder_embedding(src)
        src_pos = self.encoder_positional_encoding(torch.arange(T_src, device=device))
        src_input = src_embedding + src_pos.unsqueeze(0)
        encoder_output = self.encoder(src_input, src_key_padding_mask)

        # Start with SOS token
        generated = torch.full((B, 1), self.config.SOS, dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(max_length - 1):
                T_gen = generated.size(1)
                dec_embedding = self.decoder_embedding(generated)
                dec_pos = self.decoder_positional_encoding(torch.arange(T_gen, device=device))
                dec_input = dec_embedding + dec_pos.unsqueeze(0)

                # Causal mask
                tgt_mask = torch.triu(torch.ones(T_gen, T_gen, device=device), diagonal=1).bool()

                # No padding in generated sequence yet
                tgt_key_padding_mask = torch.zeros_like(generated, dtype=torch.bool)

                dec_output = self.decoder(dec_input, encoder_output, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
                logits = self.output_projection(dec_output[:, -1:, :])
                next_token = torch.argmax(logits, dim=-1)
                generated = torch.cat([generated, next_token], dim=1)
                if (next_token == self.config.EOS).all():
                    break
        self.train()
        return generated