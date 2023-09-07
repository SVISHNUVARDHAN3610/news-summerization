import nltk
import datasets
import numpy as np
import random
import pandas as pd
import time
import re
import csv

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt

from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

from Networks import Encoder,Decoder
from model import Seq2Seq

# #downloading data
# nltk.download('punkt')
# nltk.download('stopwords')

dataset_name = "xsum"
dataset = load_dataset(dataset_name,"all")

df = pd.DataFrame(dataset['train'])

def preprocessing(data):
  finals = []
  pattern = r'[^\w\s]|[\d]+'
  for sent in data:
    doc = ""
    doc += "<sos>"
    doc += " "
    sent = re.sub(pattern,'',sent).lower()
    for word in word_tokenize(sent):
      if word not in stopwords.words("english"):
        doc+= word
        doc+= " "
      else:
        pass
    doc += "<eos>"
    finals.append(doc)
  return finals

def tokenization(data):
  finals = []
  for sent in data:
    finals.append(word_tokenize(sent))
  return finals

def vectorization(data):
  final_dict = {}
  reverse_dict = {}
  for idx,word in enumerate(data):
    final_dict.update({word:idx})
    reverse_dict.update({idx:word})
  return final_dict,reverse_dict


src = df["document"]
trg = df["summary"]

def saving_document(document):
  with open("document.csv",'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Document"])
    writer.writerows(document)

def saving_summary(summary):
  with open("summary.csv",'w') as f:
    writer = csv.writer(f)
    writer.writerow(["summary"])
    writer.writerows(summary)

def pre_processing(src,trg,valid = True):
  if valid==True:
    print("reading document.....")
    src = pd.read_csv("document.csv",on_bad_lines='skip')
    print("reading summary.....")
    trg = pd.read_csv("summary.csv",on_bad_lines='skip')
  else:
    src = preprocessing(src)
    trg = preprocessing(trg)
  return src,trg


print("started")
start = time.time()
src,trg = pre_processing(src,trg,valid=False)
saving_document(src)
saving_summary(trg)
print("saving........")
print(f'it took {time.time()-start} seconds time to complete preprocessing')

start = time.time()
src_tokens = tokenization(src)
trg_tokens = tokenization(trg)
print(f'it took {time.time()-start} seconds time to complete tokenization')

start = time.time()
src_vectors = Word2Vec(src_tokens,min_count=1,vector_size = 64)
trg_vectors = Word2Vec(trg_tokens,min_count=1,vector_size = 64)

src_vectors,src_reverse_vectors = vectorization(list(src_vectors.wv.key_to_index))
trg_vectors,trg_reverse_vectors = vectorization(list(trg_vectors.wv.key_to_index))
print(f'it took {time.time()-start} seconds time to complete vectorization')


src_vectors.update({"<sos>":list(src_vectors.values())[-1]+1})
src_vectors.update({"<eos>":list(src_vectors.values())[-1]+1})
trg_vectors.update({"<sos>":list(trg_vectors.values())[-1]+1})
trg_vectors.update({"<eos>":list(trg_vectors.values())[-1]+1})
trg_reverse_vectors.update({list(trg_reverse_vectors.keys())[-1]+1:"<sos>"})
trg_reverse_vectors.update({list(trg_reverse_vectors.keys())[-1]+1:"<eos>"})

print("vocab_size",list(src_vectors)[-1])
print("vocab_size",list(trg_vectors)[-1])

class Main:
  def __init__(self):
    self.vocab_size    = list(src_vectors.values())[-1]+1
    self.output_size   = list(trg_vectors.values())[-1]
    self.embed_size    = 128
    self.num_layers    = 12
    self.hidden_size   = 64
    self.training_size = len(src)
    self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.encoder       = Encoder(self.embed_size,self.vocab_size,self.hidden_size,self.num_layers,self.device)
    self.decoder       = Decoder(self.embed_size,self.vocab_size,self.hidden_size,self.num_layers,self.output_size,self.device)
    self.seq2seq       = Seq2Seq(self.encoder,self.decoder,self.device,True)

    self.encoder_optim = optim.Adam(self.encoder.parameters(),lr = 0.00089)
    self.decoder_optim = optim.Adam(self.decoder.parameters(),lr = 0.00459)

    self.loss = nn.CrossEntropyLoss()

    self.org_loss      = []
    self.cross_loss    = []

    self.csv           = []
  def train(self,src,trg):
    for i in range(5121,self.training_size):
      x,y = self.tokenization(src[i],trg[i])
      outputs,sent = self.seq2seq(x,y)
      cross_loss = self.loss(outputs,y.float())
      loss = self.loss_calculate(outputs,y)
      self.org_loss.append(loss.cpu().numpy())
      self.cross_loss.append(loss.cpu().numpy()/10000000)

      loss.requires_grad = True

      self.encoder_optim.zero_grad()
      self.decoder_optim.zero_grad()
      loss.backward()
      self.encoder_optim.step()
      self.decoder_optim.step()

      self.loadandsave()
      self.ploting()
      self.csv_maker(i,loss.item(),src[i],trg[0],self.translation(outputs.long()),sent)

      print(f'episode: {i} loss: {loss.item()} document-count: {len(src[i])} summary-count: {len(trg[i])}')

  def translation(self,data):
    sent = ""
    for word in data:
      sent += trg_reverse_vectors[word.item()]
      sent += " "
    return sent

  def tokenization(self,src,trg):
    src_tokens = src.split(" ")
    trg_tokens = trg.split(" ")
    src_finals = []
    trg_finals = []

    for word in src_tokens:
      if word != " ":
        src_finals.append(src_vectors[word])

    for word in trg_tokens:
      if word != " ":
        trg_finals.append(trg_vectors[word])

    src_finals = torch.tensor(src_finals).to(self.device).long()
    trg_finals = torch.tensor(trg_finals).long().to(self.device)
    return src_finals,trg_finals

  def loadandsave(self,load=True):
    if load:
      self.encoder.load_state_dict(torch.load("Buffer/encoder.pth"))
      self.decoder.load_state_dict(torch.load("Buffer/decoder.pth"))

    torch.save(self.encoder.state_dict(),"Buffer/encoder.pth")
    torch.save(self.decoder.state_dict(),"Buffer/decoder.pth")

  def csv_maker(self,i,loss,src,trg,translation,sent):
    def teacher_force(data):
      force,normal = 0,0
      tokens = data.split(" ")
      for word in tokens:
        if word =="force":
          force +=1
        else:
          normal +=1
      return {force:normal}

    entry = [i,loss,src,trg,translation,len(src.split(" ")),len(trg.split(" ")),teacher_force(sent),sent]
    self.csv.append(entry)
    head = ['episode',"loss","document","summery","outputs","document_length","summery_length","force:normal","teacher-force"]
    if i%10==0:
      with open("main.csv",'w') as f:
        writer = csv.writer(f)
        writer.writerow(head)
        writer.writerows(self.csv)

  def ploting(self):
    plt.plot(self.org_loss)
    plt.xlabel("episodes")
    plt.ylabel("loss")
    plt.title("text-summerizatio loss")
    plt.savefig("Buffer/Normal-loss.png")
    plt.close()

    plt.plot(self.cross_loss)
    plt.xlabel("episodes")
    plt.ylabel("loss")
    plt.title("text-summerizatio loss")
    plt.savefig("Buffer/Cross-entropy-loss.png")
    plt.close()

  def loss_calculate(self,x,y):
    loss = y-x
    loss = loss.sum()
    return loss/len(x)
  

main = Main()
main.train(src,trg)