import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)   
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x): 
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, attn_mask=None):
        batch_size = q.size(0)
        q = self.q_proj(q).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, src_mask=None):
        norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(norm, norm, norm, attn_mask=src_mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.enc_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(norm, norm, norm, attn_mask=tgt_mask))
        x = x + self.dropout(self.enc_attn(self.norm2(x), enc_out, enc_out, attn_mask=src_mask))
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        x = self.embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, d_ff=1024,
                 n_layers=3, dropout=0.1, max_len=512, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = Encoder(vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

        self.encoder.embed.weight = self.decoder.embed.weight
        self.out_proj.weight = self.decoder.embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)  

    def make_tgt_mask(self, tgt):
        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2) 
        seq_len = tgt.size(1)
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt.device))
        return pad_mask & causal 

    def encode(self, src):
        return self.encoder(src, self.make_src_mask(src))

    def decode(self, tgt, enc_out, src_mask):
        return self.decoder(tgt, enc_out, self.make_tgt_mask(tgt), src_mask)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask,src_mask)
        return self.out_proj(dec_out)