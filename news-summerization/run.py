import pandas as pd
import time
import random
import csv
import re
import nltk
from main import Main
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec



#downloading data
nltk.download('punkt')
nltk.download('stopwords')


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

src = df["document"][:100]
trg = df["summary"][:100]

print("started")
start = time.time()
src = preprocessing(src)
trg = preprocessing(trg)
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


if __name__=="__main__":
    main = Main(src,trg,src_vectors,trg_vectors,trg_reverse_vectors,src_tokens,trg_tokens)
    main.train()