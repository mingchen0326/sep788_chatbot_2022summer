# -*- coding: UTF-8 -*-
# Sequence to Sequence Model

# requires
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

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

# read data


pass