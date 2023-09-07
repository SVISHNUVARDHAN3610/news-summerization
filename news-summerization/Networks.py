import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


class Encoder(nn.Module):
  def __init__(self,embed_size,vocab_size,hidden_size,num_layers,device):
    super(Encoder,self).__init__()
    self.embed_size = embed_size
    self.vocab_size = vocab_size
    self.hidden_size= hidden_size
    self.num_layers = num_layers
    self.device     = device
    self.embed_layer= nn.Embedding(self.vocab_size,self.embed_size).to(self.device)
    self.lstm       = nn.LSTM(self.embed_size,hidden_size = self.hidden_size,num_layers = self.num_layers).to(self.device)
    self.dropout    = nn.Dropout(0.5)
    self.init_weights()

  def forward(self,x):
    x = self.embed_layer(x)
    x = self.dropout(x)
    x,(hidden,cell) = self.lstm(x)
    return hidden,cell

  def init_weights(self):
    for m in self.modules():
      if isinstance(m,nn.Embedding):
        nn.init.xavier_uniform_(m.weight)
      elif isinstance(m,nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            
    
class Decoder(nn.Module):
  def __init__(self,embed_size,vocab_size,hidden_size,num_layers,output_size,device):
    super(Decoder,self).__init__()
    self.embed_size = embed_size
    self.vocab_size = vocab_size
    self.hidden_size= hidden_size
    self.num_layers = num_layers
    self.output_size= output_size
    self.device     = device
    self.embed_layer= nn.Embedding(self.vocab_size,self.embed_size).to(self.device)
    self.lstm       = nn.LSTM(self.embed_size,hidden_size = self.hidden_size,num_layers = self.num_layers).to(self.device)
    self.dropout    = nn.Dropout(0.5)
    self.init_weights()

  def forward(self,x,hidden,cell):
    x = self.embed_layer(x)
    x = self.dropout(x)
    x,(hidden,cell) = self.lstm(x,(hidden,cell))
    return x,hidden,cell

  def init_weights(self):
    for m in self.modules():
      if isinstance(m,nn.Embedding):
        nn.init.xavier_uniform_(m.weight)
      elif isinstance(m,nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
      elif isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight)


