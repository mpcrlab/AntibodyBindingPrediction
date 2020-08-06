
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import csv
from PIL import Image
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



MODEL_LOAD_DIR = '' + '/'

NUM_CLASSES = 2
INPUT_DIM = 27
DROPOUT = .5
LR = .0001
HD = 300
NL = 1
BS = 1

###############################################################################
def hot_prots(X):
    X_bin = []
    ide = np.eye(INPUT_DIM, INPUT_DIM)
    for i in range(len(X)):
        x_ = X[i]
        x = ide[x_.astype(int),:]
        X_bin.append(x)
    return X_bin

def load_batch(x, y):
    X_lengths = [x.shape[0]]
    x = np.expand_dims(x, axis=0)
    labels = torch.from_numpy(y).to(torch.long)
    ins = torch.from_numpy(np.asarray(x)).to(torch.float)
    return ins, labels, X_lengths

def cnn_batch(x,y,phase):
    ins = []
    batch_idx = np.random.choice(len(x), 1)
    batch_bin = [x[i] for i in batch_idx]
    longest = 1750
    for im in batch_bin:
        ad = np.zeros((longest-im.shape[0], 27))
        ad[:, 26] += 1
        im = np.concatenate((im, ad), axis=0)
        ins.append(im)
    labels = torch.from_numpy(y[batch_idx]).to(torch.long)
    ins = torch.from_numpy(np.asarray(ins)).to(torch.float)
    return ins, labels

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
xtrain = np.load('data/bind/bind_train.npy')
ntrain = np.load('data/notbind/nobind_train.npy')
xtest = np.load('data/bind/bind_test.npy')
ntest = np.load('data/notbind/nobind_test.npy')
xtrain = [np.flip(i) for i in xtrain]
ntrain = [np.flip(i) for i in ntrain]
xtest = [np.flip(i) for i in xtest]
ntest = [np.flip(i) for i in ntest]
X = np.append(xtrain, xtest)
NX = np.append(ntrain, ntest)
bhot_seqs = hot_prots(X)
nhot_seqs = hot_prots(NX)
print('# of bind test sequences:', len(bhot_seqs))
print('# of not-bind test sequences:', len(nhot_seqs))

###

class ConvNet(nn.Module):
    def __init__(self, classes = 2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2),
            nn.Dropout(.5),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 183, kernel_size=5, stride=1, padding=1),
            nn.Dropout(.5),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(183, 1, kernel_size=1, stride=1, padding=1),
            nn.Dropout(.5),
            nn.ReLU())
        self.fc = nn.Linear(50808, 2)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class Classifier_LSTM(nn.Module):
    def __init__(self, NL, HD, BS):
        super(Classifier_LSTM, self).__init__()
        self.NL = NL
        self.HD = HD
        self.BS = BS
        self.lstm1 =  nn.LSTM(INPUT_DIM, self.HD, num_layers=self.NL, bias=True, BStch_first=True)
        self.drop = nn.Dropout(p=DROPOUT)
        self.fc = nn.Linear(HD, NUM_CLASSES)
        self.sig = nn.Sigmoid()
    def forward(self, inputs, X_lengths, hidden):
        X, hidden1 = self.lstm1(inputs)
        X = X[:,-1,:]
        out = self.drop(X)
        out = self.fc(X)
        out = self.sig(out)
        return out, hidden1
    def init_hidden1(self, NL, BS):
        weight = next(model.parameters()).data
        hidden1 = (weight.new(NL, BS, HD).zero_().to(torch.int64),
                  weight.new(NL, BS, HD).zero_().to(torch.int64))
        return hidden1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

################################################################

def test(data, y):
    bpbin = {}
    names = []
    idx = 0
    for model_path in os.listdir(MODEL_LOAD_DIR):
        if '.pt' in model_path:
            corrects = []
            name = MODEL_LOAD_DIR + model_path
            names.append(model_path.split(']')[0])
            model = torch.load(name)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            model.eval()
            if LSTM == True:
                h1 = model.init_hidden1(1, 1)
            for input in data:
                optimizer.zero_grad()
                if LSTM == True:
                    inputs, labels, X_lengths = load_batch(input, y)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outs, h = model(inputs, X_lengths, h1)
                if CNN == True:
                    input = input[None,:,:]
                    inputs, labels = cnn_batch(input,y, 'val')
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs[None,:,:,:]
                    outs = model(inputs)
                _, preds = outs.max(1)
                if preds.item() == 1:
                    corrects.append(1)
                elif preds.item() == 0:
                    corrects.append(0)
            p = np.asarray(corrects).sum()
            bpbin.update({idx:p})
            idx += 1
            print(name)
            print('# of test sequences correctly predicted:', p)
    bavg = np.mean(np.asarray(list(bpbin.values()))/len(bhot_seqs))
    bindrank = {k:v for k,v in sorted(bpbin.items(), key=lambda item:item[1])}
    bindbest = {}
    for k,v in  list(bindrank.items()):
        if v/len(bhot_seqs) >= 0.95:
            bindbest.update({names[k]:v})
    print('- ALPHA-MODEL:', bindbest)
    print('- Average correct across models:', bavg)


def both_test(notbpbin, bpbin):
    apbin = np.stack((np.asarray(list(bpbin.values())), np.asarray(list(notbpbin.values()))))
    s = np.sum(apbin, axis=0)/(len(bhot_seqs)*2)
    plt.close()
    plt.bar(np.arange(len(s)), s)
    plt.title('Correct All Sequence Predictions by Trained Models (Flipped)')
    plt.show()
    both = {}
    for idx, m in enumerate(s):
        if m > .95:
            both.update({names[idx]: m})
    b = {k:v for k,v in sorted(both.items(), key=lambda item:item[1])}
    print('\n', 'Both-Alpha-Models:', b)
    print('Both-avg:', np.average(s))




#
