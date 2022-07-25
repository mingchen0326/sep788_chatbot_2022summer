# -*- coding: UTF-8 -*-
# Sequence to Sequence Model

# requires
import sys
import time
import os
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import seq2seq_model as seq2seqModel

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load files
train_data_raw = pd.read_csv("Dataset/train_df.csv")
eval_data_raw = pd.read_csv("Dataset/eval_df.csv")
test_data_raw = pd.read_csv("Dataset/test_df.csv")

# model data structure
SOS_token = 0   # start of sentence
EOS_token = 1   # end of sentence

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# preprocess data
def preprocess_sentence(w):
    w ='start '+ w + ' end'
    #print(w)
    return w

def create_dataset(data):
    pairs = data.loc[:,'Question': "Answer"].values.tolist()
    pairs_len = len(pairs)
    i = 0
    while i < pairs_len:
        pairs[i] = [preprocess_sentence(pairs[i][0]),preprocess_sentence(pairs[i][1])]
        i += 1

    input_lang=Lang("ans")
    output_lang=Lang("ask")
    pairs = [list(reversed(p)) for p in pairs]
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang,output_lang,pairs

# read data
def read_data(data):
    input_tensors=[]
    target_tensors=[]
    input_lang,target_lang,pairs=create_dataset(data)

    for i in range(len(pairs)-1):
        input_tensor = tensorFromSentence(input_lang, pairs[i][0])
        target_tensor = tensorFromSentence(target_lang, pairs[i][1])
        input_tensors.append(input_tensor)
        target_tensors.append(target_tensor)
    return input_tensors,input_lang,target_tensors,target_lang

input_tensor,input_lang,target_tensor,target_lang= read_data(train_data_raw)
hidden_size = 256

# train model
def train(BATCH_SIZE=1, max_data=50000, min_loss=0.7):
    steps_per_epoch = len(input_tensor) // BATCH_SIZE
    print(steps_per_epoch)
    checkpoint_dir = 'model_data'

    checkpoint_prefix = os.path.join(checkpoint_dir, ".pt")
    start_time = time.time()
    encoder = seq2seqModel.Encoder(input_lang.n_words, hidden_size).to(device)
    decoder = seq2seqModel.AttentionDencoder(hidden_size, target_lang.n_words, dropout_p=0.1).to(device)
    if os.path.exists(checkpoint_prefix):
        checkpoint = torch.load(checkpoint_prefix)
        encoder.load_state_dict(checkpoint['modelA_state_dict'])
        decoder.load_state_dict(checkpoint['modelB_state_dict'])

    total_loss = 0
    batch_loss=1
    while batch_loss>min_loss:
        start_time_epoch = time.time()
        for i in range(1,(max_data//BATCH_SIZE)):
            inp=input_tensor[(i-1)*BATCH_SIZE:i*BATCH_SIZE]
            targ=target_tensor[(i-1)*BATCH_SIZE:i*BATCH_SIZE]
            batch_loss = seq2seqModel.train_step(inp, targ,encoder,decoder,optim.SGD(encoder.parameters(),lr=0.001),optim.SGD(decoder.parameters(),lr=0.01))
            total_loss += batch_loss
            print('训练总步数:{} 最新每步loss {:.4f}'.format(i,batch_loss ))
        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch
        current_steps = +steps_per_epoch
        step_time_total = (time.time() - start_time) / current_steps
        print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(current_steps, step_time_total, step_time_epoch,
                                                                      batch_loss))
        torch.save({'modelA_state_dict': encoder.state_dict(),
                     'modelB_state_dict': decoder.state_dict()},checkpoint_prefix)
        sys.stdout.flush()

pass

read_data(train_data_raw)