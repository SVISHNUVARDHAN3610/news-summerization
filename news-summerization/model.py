import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import random


class Seq2Seq(nn.Module):
  def __init__(self,encoder,decoder,device,teacher_force):
    super(Seq2Seq,self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.teacher_force = teacher_force
    self.force         = 0.345
    self.device        = device

  def forward(self,src,trg):
    sent =' '
    outputs = []
    hidden,cell = self.encoder(src)
    x = torch.tensor(trg[0]).long().to(self.device)
    x = x.unsqueeze(0)
    for i in range(len(trg)):
      x,hidden,cell = self.decoder(x,hidden,cell)
      if self.teacher_force:
        if random.uniform(0,1)>self.force:
          x = trg[i]
          x = x.unsqueeze(0)
          sent+= "force"
          sent += " "
        else:
          x = x.argmax(1)
          sent+= "argmax"
          sent += " "
      outputs.append(x)
    outputs = torch.tensor(outputs).float().to(self.device)
    return outputs,sent