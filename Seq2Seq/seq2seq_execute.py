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
from tqdm import tqdm

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load files
train_data_raw = pd.read_csv("Dataset/train_df.csv")
eval_data_raw = pd.read_csv("Dataset/eval_df.csv")
test_data_raw = pd.read_csv("Dataset/test_df.csv")

# model data structure
SOS_token = 0   # start of sentence
EOS_token = 1   # end of sentence
MAX_LENGTH = 30


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

    def addWord_pred(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1

    def showWords(self):
        words = self.word2index.keys()
        print(words)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# preprocess data


def preprocess_sentence(w):
    return w


def create_dataset(data):
    pairs = data.loc[:, 'Question': "Answer"].values.tolist()
    pairs_len = len(pairs)
    i = 0
    while i < pairs_len:
        pairs[i] = [preprocess_sentence(
            pairs[i][0]), preprocess_sentence(pairs[i][1])]
        i += 1

    input_lang = Lang("ask")
    output_lang = Lang("ans")

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang, output_lang, pairs

# read data


def read_data(data):
    input_tensors = []
    target_tensors = []
    input_lang, target_lang, pairs = create_dataset(data)

    for i in range(len(pairs)-1):
        input_tensor = tensorFromSentence(input_lang, pairs[i][0])
        target_tensor = tensorFromSentence(target_lang, pairs[i][1])
        input_tensors.append(input_tensor)
        target_tensors.append(target_tensor)
    return input_tensors, input_lang, target_tensors, target_lang


input_tensor, input_lang, target_tensor, target_lang = read_data(
    train_data_raw)
hidden_size = 256

# train model


def train(BATCH_SIZE=1, max_data=50000, min_loss=0.2):
    print("training...")
    steps_per_epoch = len(input_tensor) // BATCH_SIZE
    print(steps_per_epoch)
    checkpoint_dir = 'Seq2Seq'

    if max_data > len(input_tensor):
        max_data = len(input_tensor)

    checkpoint_prefix = (checkpoint_dir + "/model.pt")
    start_time = time.time()
    encoder = seq2seqModel.Encoder(input_lang.n_words, hidden_size).to(device)
    decoder = seq2seqModel.AttentionDencoder(
        hidden_size, target_lang.n_words, dropout_p=0.1).to(device)
    if os.path.exists(checkpoint_prefix):
        checkpoint = torch.load(checkpoint_prefix)
        encoder.load_state_dict(checkpoint['modelA_state_dict'])
        decoder.load_state_dict(checkpoint['modelB_state_dict'])

    total_loss = 0
    batch_loss = 1
    while batch_loss > min_loss:
        start_time_epoch = time.time()
        for i in tqdm(range(1, (max_data//BATCH_SIZE))):
            inp = input_tensor[(i-1)*BATCH_SIZE:i*BATCH_SIZE]
            targ = target_tensor[(i-1)*BATCH_SIZE:i*BATCH_SIZE]
            batch_loss = seq2seqModel.train_step(inp, targ, encoder, decoder, optim.SGD(
                encoder.parameters(), lr=0.001), optim.SGD(decoder.parameters(), lr=0.01))
            total_loss += batch_loss

        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch
        current_steps = +steps_per_epoch
        step_time_total = (time.time() - start_time) / current_steps
        print('Total Step: {} Time per Step: {}  Last Step Time: {} Last Step Loss {:.4f}'.format(current_steps, step_time_total, step_time_epoch,
                                                                      batch_loss))
        torch.save({'modelA_state_dict': encoder.state_dict(),
                    'modelB_state_dict': decoder.state_dict()}, checkpoint_prefix)
        sys.stdout.flush()


def predict(sentence, model_path='Seq2Seq/model.pt'):
    max_length_tar = MAX_LENGTH
    encoder = seq2seqModel.Encoder(input_lang.n_words, hidden_size).to(device)
    decoder = seq2seqModel.AttentionDencoder(
        hidden_size, target_lang.n_words, dropout_p=0.1).to(device)
    checkpoint_prefix = model_path
    checkpoint = torch.load(checkpoint_prefix)
    encoder.load_state_dict(checkpoint['modelA_state_dict'])
    decoder.load_state_dict(checkpoint['modelB_state_dict'])

    # input_lang.showWords()

    sentence = preprocess_sentence(sentence)
    sentence_list = []
    for word in sentence.split(' '):            # remove words not in dict
        if word in input_lang.word2index:
            sentence_list.append(word)
    
    sentence_list = sentence_list[:MAX_LENGTH-1]        # cut sentence to max length
    sentence = ' '.join(sentence_list)
    input_tensor = tensorFromSentence(input_lang, sentence)

    input_length = input_tensor.size()[0]
    result = ''
    max_length = MAX_LENGTH
    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]

    dec_input = torch.tensor([[SOS_token]], device=device)  # SOS

    dec_hidden = encoder_hidden
    #decoder_attentions = torch.zeros(max_length, max_length)
    for t in range(max_length_tar):
        predictions, dec_hidden, decoder_attentions = decoder(
            dec_input, dec_hidden, encoder_outputs)
        predicted_id, topi = predictions.data.topk(1)

        if topi.item() == EOS_token:
            break
        else:
            result += target_lang.index2word[topi.item()]+' '
        dec_input = topi.squeeze().detach()
    # print(result)
    return result

def pred_datas(data, file_name):
    input_data = data.loc[:, 'Question'].values.tolist()
    full_data = data.loc[:, 'Question': "Answer"].values.tolist()
    results = []
    i = 0
    print("predicting...")
    for ask in tqdm(input_data):
        result = predict(ask)
        full_data[i] = full_data[i] + [result]
        i += 1
    full_data_df = pd.DataFrame(full_data)
    full_data_df.columns = ['Question', 'Answer', 'Predictive Answer']
    full_data_df.to_csv("Results/%s" % (file_name), index=False)
    print("result save to %s" % (file_name))
    pass


# code entry

# train()
# predict('do female polar bears weight more than the male')
pred_datas(train_data_raw, "seq2seq_result_train.csv")
pred_datas(test_data_raw, "seq2seq_result_test.csv")
