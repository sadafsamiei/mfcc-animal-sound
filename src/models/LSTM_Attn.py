import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAttention(nn.Module):

    def __init__(self, d_model: int, d_attn: int = 64):
        super().__init__()
        self.proj = nn.Linear(d_model, d_attn, bias=True)
        self.v = nn.Linear(d_attn, 1, bias=False)  

    def forward(self, H):
        scores = self.v(torch.tanh(self.proj(H))).squeeze(-1) 
        alpha = torch.softmax(scores, dim=1)                    
        context = torch.bmm(alpha.unsqueeze(1), H).squeeze(1) 
        return context, alpha


class MFCC_LSTM_Attn(nn.Module):

    def __init__(self, n_mfcc: int, num_classes: int, hidden: int = 128, num_layers: int = 1, bidir: bool = True, dropout: float = 0.1):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.lstm = nn.LSTM(
            input_size=n_mfcc,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidir,
            batch_first=True,
        )
        d_model = hidden * (2 if bidir else 1)
        self.attn = GlobalAttention(d_model, d_attn=64)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1).transpose(1, 2) 
        H, _ = self.lstm(x)                  
        ctx, alpha = self.attn(H)             
        logits = self.head(self.dropout(ctx)) 
        return logits, alpha
