import torch.nn as nn 
from dataclasses import dataclass 
import torch 

@dataclass 
class config:
    d_model = 512 
    n_heads = 8 
    N = 6 
    context_length = 1000 
    int_dim = 2048 


class FFN(nn.Module):
    def __init__(self,d_model,int_dim):
        super().__init__()
        self.layer1 = nn.Linear(d_model,int_dim)
        self.layer2 = nn.Linear(int_dim,d_model)
        self.act = nn.ReLU()

    def forward(self,idx:torch.Tensor):
        return self.layer2(self.act(self.layer1(idx))) # return shape B,T,C 

class TransformerEncoderLayer(nn.Module): 
    def __init__(self, d_model, nhead,int_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,batch_first=True)
        self.norm1 = nn.LayerNorm(d_model) 
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FFN(d_model=d_model, int_dim=int_dim)
    def forward(self,idx:torch.Tensor):
        B,T,C = idx.shape 
        x , _ = self.self_attn(idx,idx,idx)
        x = self.norm1(idx + x)
        x = self.norm2(x + self.feed_forward(x))
        return x 



class Encoder(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config 
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.int_dim = config.int_dim 
        self.N = config.N 
        self.context_length = config.context_length 
        self.layers = nn.ModuleList([TransformerEncoderLayer(self.d_model,self.n_heads,self.int_dim) for _ in range(self.N)]) 
    
    def forward(self, x:torch.Tensor): 
        B,T,C = x.shape 
        assert C == self.d_model, f"Input tensor must have shape (B,T,{self.d_model})"
        for layer in self.layers:
            x = layer(x)
        return x 
     


class Decoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config 
        self.d_model = config.d_model 
        self.n_heads = config.n_heads 
        self.int_dim = config.int_dim 
        self.N = config.N 
        self.layers = ... # continue from here 
        self.context_length = config.context_length 
    

    def forward(self,x):
        return x 



class Translator(nn.Module):
    def __init__(self,config = config()) -> None:
        super().__init__() 
        self.config = config 
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.N = config.N
        self.context_length = config.context_length 
        self.encoder = Encoder(config)  
        self.decoder = Decoder(config)
        self.linear = nn.Linear(self.d_model,self.d_model)
        self.encoder_embedding = nn.Embedding(self.context_length, self.d_model)
        self.decoder_embedding = nn.Embedding(self.context_length, self.d_model) 
        self.encoder_positional_encoding = nn.Embedding(self.context_length, self.d_model)
    
    def forward(self,idx:torch.Tensor):
        B, T = idx.shape 
        assert T <= self.context_length, f"Input tensor must have shape (B,T) with T <= {self.context_length}" 
        input_embedding = self.encoder_embedding(idx)
        input_positional_encoding = self.encoder_positional_encoding(torch.arange(T, device=idx.device))   # shape (T, d_model)
        input_tensor = input_embedding + input_positional_encoding.unsqueeze(0)   # shape (B, T, d_model) . this will be passed to encoder so that we can get the output of encoder
        encoder_output = self.encoder(input_tensor) 
        decoder_output = self.decoder(encoder_output)  # shape B,T,d_model 
        return self.linear(decoder_output) # these are output logits you need to apply sotmax for probability 
    