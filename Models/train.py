                                                                                                                                                                                                                                                                                                                                          # -*- coding: utf-8 -*-
"""LSTM-pytorch
"""


import csv
import time
import copy
import random
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# torch.manual_seed(0)
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark=False

# np.random.seed(0)

import wandb
wandb.init(project='antibodies-flipped-CNN_sig')
wab = wandb.config

LSTM = False

wab.NUM_CLASSES = 2
wab.N_LAYERS = 1
wab.INPUT_DIM = 27
wab.HIDDEN_DIM = 200
wab.DROPOUT = .5
wab.LR = .0001
wab.BS = 1
wab.NUM_EPOCHS = 200
wab.OPTIM = 'adam'



rando = random.randint(0, 100000)
wab.RANDO = rando


RESULTS = 'results/CNN_sig/'
PRESAVE_NAME = RESULTS + ('/CNN-'+str(rando)+'--'+str(wab.NUM_EPOCHS)+'e-'+str(wab.LR)+'lr-'+str(wab.BS)+'bs-'+str(wab.HIDDEN_DIM)+'hd-'+str(wab.OPTIM)+'opt-')


#some stuff for wandb
nl = wab.N_LAYERS
hd = wab.HIDDEN_DIM
ba = wab.BS



"""load the data"""

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
xtest = np.load('data/bind/bind_test.npy')
xtrain = np.load('data/bind/bind_train.npy')
ntrain = np.load('data/notbind/nobind_train.npy')
ntest = np.load('data/notbind/nobind_test.npy')
xtestf = [np.flip(i) for i in xtest]
xtrainf = [np.flip(i) for i in xtrain]
ntestf = [np.flip(i) for i in ntest]
ntrainf = [np.flip(i) for i in ntrain]
ybtest = np.ones(len(xtestf))
ybtrain = np.ones(len(xtrainf))
yntrain = np.zeros(len(ntrainf))
yntest = np.zeros(len(ntestf))
testx = np.append(xtestf, ntestf)
trainx = np.append(xtrainf, ntrainf)
trainy = np.append(ybtrain, yntrain)
testy = np.append(ybtest, yntest)

##if you wanna shuffle the labels--->
# rng = np.random.default_rng()
# rng.shuffle(trainy)

def hot_prots(X):
    X_bin = []
    ide = np.eye(wab.INPUT_DIM, wab.INPUT_DIM)
    for i in range(X.shape[0]):
        x_ = X[i]
        x = ide[x_.astype(int),:]
        X_bin.append(x)
    return X_bin

def cnn_batch(x,y,phase):
    ins = []
    batch_idx = np.random.choice(len(x), wab.BS)
    batch_bin = [x[i] for i in batch_idx]
    longest = 1750
    for im in batch_bin:
        ad = np.zeros((longest-im.shape[0], wab.INPUT_DIM))
        ad[:, 26] += 1
        im = np.concatenate((im, ad), axis=0)
        ins.append(im)
    labels = torch.from_numpy(y[batch_idx]).to(torch.long)
    ins = torch.from_numpy(np.asarray(ins)).to(torch.float)
    return ins, labels

def load_batch(x, y, phase):
    ins = []
    batch_idx = np.random.choice(len(x), wab.BS)
    batch_bin = [x[i] for i in batch_idx]
    X_lengths = [im.shape[0] for im in batch_bin]
    longest = max(X_lengths)
    for im in batch_bin:
        ad = np.zeros((longest-im.shape[0], wab.INPUT_DIM))
        ad[:, 26] += 1
        im = np.concatenate((im, ad), axis=0)
        ins.append(im)
    labels = torch.from_numpy(y[batch_idx]).to(torch.long)
    ins = torch.from_numpy(np.asarray(ins)).to(torch.float)
    return ins, labels, X_lengths


X = hot_prots(trainx)
Xt = hot_prots(testx)
xtrainf = [np.flip(i, axis=1) for i in X]
xtestf = [np.flip(i, axis=1) for i in Xt]
trainy = np.append(trainy, trainy)
testy = np.append(testy, testy)
X = np.append(X, xtrainf)
Xt = np.append(Xt, xtestf)
print('Training Size:', len(X), 'Test Size:', len(Xt))
data = {'train': [X, trainy], 'test': [Xt, testy]}



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class ConvNet(nn.Module):
    def __init__(self, classes = wab.NUM_CLASSES):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2),
            nn.Dropout(wab.DROPOUT),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 183, kernel_size=5, stride=1, padding=1),
            nn.Dropout(wab.DROPOUT),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(183, 1, kernel_size=1, stride=1, padding=1),
            nn.Dropout(wab.DROPOUT),
            nn.ReLU())
        self.fc = nn.Linear(50808, 2)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.sig(out)
        return out

class Classifier_LSTM(nn.Module):
    def __init__(self, nl, hd, ba):
        super(Classifier_LSTM, self).__init__()
        self.nl = nl
        self.hd = hd
        self.ba = ba
        self.lstm1 =  nn.LSTM(wab.INPUT_DIM, hd, num_layers=nl, bias=True, batch_first=True)
        self.relu = nn.RReLU()
        self.drop = nn.Dropout(p=wab.DROPOUT)
        self.fc = nn.Linear(hd, wab.NUM_CLASSES)
        self.sig = nn.Sigmoid()
    def forward(self, inputs, X_lengths, hidden):
        X, hidden1 = self.lstm1(inputs)
        X = X[:,-1,:]
        out = self.relu(X)
        out = self.drop(X)
        out = self.fc(X)
        out = self.sig(out)
        return out, hidden1
    def init_hidden1(self, nl, ba):
        weight = next(model.parameters()).data
        hidden1 = (weight.new(nl, ba, hd).zero_().to(torch.int64).to(device),
                  weight.new(nl, ba, hd).zero_().to(torch.int64).to(device))
        return hidden1

if LSTM == True:
    model = Classifier_LSTM(nl, hd,ba)
else:
    model = ConvNet(wab.NUM_CLASSES)

model.to(device)


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total # Trainable Parameters = ', pytorch_total_params,  '!!!!!!!')

criterion = nn.CrossEntropyLoss()


if wab.OPTIM == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=wab.LR)
elif wab.OPTIM == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=wab.LR)



def train():
    best_acc = 0
    for epoch in range(wab.NUM_EPOCHS):
        if LSTM == True:
            h1 = model.init_hidden1(wab.N_LAYERS, wab.BS)
        for phase in ['train', 'test']:
            running_loss = 0
            running_corrects = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()
            x,y = data[phase]
            for i in range(len(x)//wab.BS):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if LSTM == True:
                        inputs, labels, X_lengths = load_batch(x,y, phase)
                        inputs, labels = inputs.to(device), labels.to(device)
                        outs, h = model(inputs, X_lengths, h1)
                    else:
                        inputs, labels = cnn_batch(x,y, phase)
                        inputs, labels = inputs.to(device), labels.to(device)
                        inputs = inputs[:,None,:,:]
                        outs = model(inputs)
                        # sys.exit()
                    _, preds = outs.max(1)
                    loss = criterion(outs, labels)
                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(x)
            epoch_acc = running_corrects.double() / len(x)
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                wandb.log({'train_acc': epoch_acc.detach().cpu().item()}, step=epoch)
                wandb.log({'train_loss': epoch_loss}, step=epoch)
            if phase == 'test':
                wandb.log({'val_acc': epoch_acc.detach().cpu().item()}, step=epoch)
                wandb.log({'val_loss': epoch_loss}, step=epoch)
    model.load_state_dict(best_model_wts)
    SAVE_NAME = PRESAVE_NAME + str(epoch_acc.detach().cpu().item()) +'.pt'
    torch.save(model, SAVE_NAME)
    print('Best val Acc: {:4f}'.format(epoch_acc.detach().cpu().item()))
    return model, best_acc


model, best = train()












#
