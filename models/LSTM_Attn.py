import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1)  
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)    
        return context, weights

class LSTM_Attn(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64, num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.attn = Attention(hidden_dim * 2)
        self.dropout = nn.Dropout(p=dropout)  
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)                 
        context, _ = self.attn(out)
        context = self.dropout(context)       
        logits = self.fc(context)
        return logits
