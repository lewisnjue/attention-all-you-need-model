import torch.nn as nn 
from dataclasses import dataclass 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
@dataclass 
class config:
    d_model = 512 
    n_heads = 8 
    N = 6 
    context_length = 1000 
    int_dim = 2048 


class FFN(nn.Module):
    def __init__(self,d_model:int,int_dim:int):
        super().__init__()
        self.layer1 = nn.Linear(d_model,int_dim)
        self.layer2 = nn.Linear(int_dim,d_model)
        self.act = nn.ReLU()

    def forward(self,idx:torch.Tensor):
        return self.layer2(self.act(self.layer1(idx))) # return shape B,T,C 

class Head(nn.Module):
    def __init__(self,h_size):
        super().__init__()
        self.h_size = h_size 
        self.sotfmax = nn.Softmax(-1)
    
    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor):
        return self.sotfmax(q @ k.T / self.h_size) * v 


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q:torch.Tensor,k:torch.Tensor,v:torch.Tensor):
        B,T,D = q.shape 


        # 2. Reshape into multiple heads
        def reshape_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # 3. Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, T, T)
        attn = F.softmax(scores, dim=-1)  # (B, num_heads, T, T)
        out = attn @ v  # (B, num_heads, T, head_dim)

        # 4. Concatenate heads and project out
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        return self.out_proj(out), None # i did this to make sure i dont get an error if i whcih to pytorch attention


        
class TransformerEncoderLayer(nn.Module): 
    def __init__(self, d_model, nhead,int_dim):
        super().__init__()
        #self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,batch_first=True)
        self.self_attn =  MultiHeadAttention(embed_dim=d_model,num_heads=nhead)
        self.norm1 = nn.LayerNorm(d_model) 
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FFN(d_model=d_model, int_dim=int_dim)
    def forward(self,idx:torch.Tensor):
        B,T,C = idx.shape 
        x , _ = self.self_attn(idx,idx,idx)
        x = self.norm1(idx + x)
        x = self.norm2(x + self.feed_forward(x))
        return x 


class TransformerDecoderLayer(nn.Module):
    def __init__(self,d_model,nhead,int_dim):
        super().__init__()
        self.self_attn = ... 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = FFN(d_model=d_model,int_dim=int_dim)

    def forward(self,idx:torch.Tensor,decoder_output: torch.Tensor):
        B,T,C = idx.shape 
        x, _ = self.self_attn(idx,decoder_output,decoder_output) # continue from here 


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
        self.layers = nn.ModuleList([TransformerDecoderLayer(self.)])
        self.context_length = config.context_length 
    

    def forward(self,x:torch.Tensor,encoder_output:torch.Tensor):
        B,T,C = x.shape 
        assert x.shape == encoder_output.shape 
        for layer in self.layers:
            x = layer(x,encoder_output)
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
        self.decoder_positional_encoding = nn.Embedding(self.context_length,self.d_model)
    
    def forward(self,idx:torch.Tensor,target:torch.Tensor,encoder_output:Union[torch.Tensor,None] = None):
        B, T = idx.shape 
        assert T <= self.context_length, f"Input tensor must have shape (B,T) with T <= {self.context_length}" 
        if not encoder_output:
            input_embedding = self.encoder_embedding(idx)
            input_positional_encoding = self.encoder_positional_encoding(torch.arange(T, device=idx.device))   # shape (T, d_model)
            input_tensor = input_embedding + input_positional_encoding.unsqueeze(0)
            encoder_output = self.encoder(input_tensor)
        

        decoder_embedding = self.decoder_embedding(idx)
        decoder_positional_encodeing = self.decoder_positional_encoding(torch.arange(T,device=idx.device))
        decoder_input_tensor = decoder_embedding + decoder_positional_encodeing  
        decoder_output = self.decoder(decoder_input_tensor,encoder_output)



    