
import os, sys
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

INSERT = False
KNOCKOUT = False
KNOCKOUT_PEAKS = True
FLIPPED = True

MODELS =  '' + '/'
MODEL_PATH = '' + '/'

NUM_CLASSES = 2
INPUT_DIM = 27
DROPOUT = .5
LR = .0001
HD = 200
NL = 1
BS = 1

##############################

def hot_prots(X):
    X_bin = []
    ide = np.eye(INPUT_DIM, INPUT_DIM)
    for i in range(len(X)):
        x_ = X[i]
        x = ide[x_.astype(int),:]
        X_bin.append(x)
    return X_bin

def load_batch(x):
    X_lengths = [x.shape[0]]
    x = np.expand_dims(x, axis=0)
    ins = torch.from_numpy(np.asarray(x)).to(torch.float)
    return ins, X_lengths

def get_data():
    xtrain = np.load('data/bind/bind_train.npy')
    ntrain = np.load('data/notbind/nobind_train.npy')
    xtest = np.load('data/bind/bind_test.npy')
    ntest = np.load('data/notbind/nobind_test.npy')
    sxtrain = xtrain.copy()
    sxtest = xtest.copy()
    sntrain = ntrain.copy()
    sntest = ntest.copy()
    if FLIPPED == True:
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
    return bhot_seqs, nhot_seqs, sxtrain, sxtest, sntrain, sntest


bhot_seqs, nhot_seqs, xtrain, xtest, ntrain, ntest = get_data()

#test sequence of interest and swapping tests
dd = ['QVKLLESGPELVKPGASVKMSCKASGYTFTSYVMHWVKQKPGQGLEWIGYINPYNDGTKYNEKFKGKATLTSDKSSSTAYMELSSLTSEDSAVYYCVRGGYRPYYAMDYWGQGTSVTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPRDCTSHHHHHHELQMTQSPASLSASVGETVTITCRASENIYSYLAWYQQKQGKSPQLLVYNAKTLAEGVPSRFSGSGSGTQFSLKINSLQPEDFGSYYCQHHYGTPLTFGAGTKLELKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNEC']
ds = []
for c in dd[0]:
    ds.append(int(ord(c.lower())-97))

ds = np.expand_dims(ds, 0)
data = hot_prots([np.flip(i) for i in ds])

allx = np.tile(ds, (81,1))
alln = np.append(ntrain, ntest)
alln = sorted(alln, key=len)


def insert_bind(alln, allx):
    for idx, s in enumerate(allx):
        alln[idx][30:36] = s[30:36]
        alln[idx][49:67] = s[49:67]
        alln[idx][98:110] = s[98:110]
        if alln[idx].shape[0] > 253:
            alln[idx][253:265] = s[253:265]
        if alln[idx].shape[0] > 278:
            alln[idx][278:287] = s[278:287]
        if alln[idx].shape[0] == 325:
            alln[idx][318:-1] = s[318:324]
        if alln[idx].shape[0] > 328:
            alln[idx][318:328] = s[318:328]
    return alln

def knockout(alln, allx):
    for idx, s in enumerate(allx):
        r = random.randint(5,81)
        s[30:35] = alln[idx][30:35]
        s[49:66] = alln[idx][49:66]
        s[98:109] = alln[idx][98:109]
        if alln[idx].shape[0] < 252:
            s[253:264] = alln[idx+r][253:264]
            s[318:327] = alln[idx+r][318:327]
        if alln[idx].shape[0] > 252:
            s[253:264] = alln[idx][253:264]
        if alln[idx].shape[0] > 277:
            s[278:286] = alln[idx][278:286]
        if alln[idx].shape[0] == 325:
            s[318:324] = alln[idx][318:-1]
        if alln[idx].shape[0] > 327:
            s[318:327] = alln[idx][318:327]
    return allx

def knockout_peaks(alln, allx, w):
    for idx, s in enumerate(allx):
        if alln[idx].shape[0]<400:
            r = random.randint(7,81)
            d = 399 - alln[idx].shape[0]
            y = np.append(alln[idx], alln[r][-d:])
            s[w] = y[w]
        else:
            s[w] = alln[idx][w]
    return allx


if INSERT == True:
    bhot_seqs, nhot_seqs, xtrain, xtest, ntrain, ntest = get_data()
    dd = ['QVKLLESGPELVKPGASVKMSCKASGYTFTSYVMHWVKQKPGQGLEWIGYINPYNDGTKYNEKFKGKATLTSDKSSSTAYMELSSLTSEDSAVYYCVRGGYRPYYAMDYWGQGTSVTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPRDCTSHHHHHHELQMTQSPASLSASVGETVTITCRASENIYSYLAWYQQKQGKSPQLLVYNAKTLAEGVPSRFSGSGSGTQFSLKINSLQPEDFGSYYCQHHYGTPLTFGAGTKLELKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNEC']
    ds = []
    for c in dd[0]:
        ds.append(int(ord(c.lower())-97))
    ds = np.expand_dims(ds, 0)
    allx = np.tile(ds, (81,1))
    alln = np.append(ntrain, ntest)
    alln = sorted(alln, key=len)
    inserts = insert_bind(alln, allx)
    X = [np.flip(i) for i in inserts]

if KNOCKOUT == True:
    bhot_seqs, nhot_seqs, xtrain, xtest, ntrain, ntest = get_data()
    dd = ['QVKLLESGPELVKPGASVKMSCKASGYTFTSYVMHWVKQKPGQGLEWIGYINPYNDGTKYNEKFKGKATLTSDKSSSTAYMELSSLTSEDSAVYYCVRGGYRPYYAMDYWGQGTSVTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPRDCTSHHHHHHELQMTQSPASLSASVGETVTITCRASENIYSYLAWYQQKQGKSPQLLVYNAKTLAEGVPSRFSGSGSGTQFSLKINSLQPEDFGSYYCQHHYGTPLTFGAGTKLELKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNEC']
    ds = []
    for c in dd[0]:
        ds.append(int(ord(c.lower())-97))
    ds = np.expand_dims(ds, 0)
    allx = np.tile(ds, (81,1))
    alln = np.append(ntrain, ntest)
    alln = sorted(alln, key=len)
    ko = knockout(alln, allx)
    X = [np.flip(i) for i in ko]

if KNOCKOUT_PEAKS == True:
    bhot_seqs, nhot_seqs, xtrain, xtest, ntrain, ntest = get_data()
    dd = ['QVKLLESGPELVKPGASVKMSCKASGYTFTSYVMHWVKQKPGQGLEWIGYINPYNDGTKYNEKFKGKATLTSDKSSSTAYMELSSLTSEDSAVYYCVRGGYRPYYAMDYWGQGTSVTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPRDCTSHHHHHHELQMTQSPASLSASVGETVTITCRASENIYSYLAWYQQKQGKSPQLLVYNAKTLAEGVPSRFSGSGSGTQFSLKINSLQPEDFGSYYCQHHYGTPLTFGAGTKLELKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNEC']
    ds = []
    for c in dd[0]:
        ds.append(int(ord(c.lower())-97))
    ds = np.expand_dims(ds, 0)
    allx = np.tile(ds, (81,1))
    alln = np.append(ntrain, ntest)
    alln = sorted(alln, key=len)
    kp = knockout_peaks(alln, allx, w)
    X = [np.flip(i) for i in kp]


sbhot_seqs = hot_prots(X)
print('# of bind test sequences:', len(sbhot_seqs))

####collect hiddens based on data and label with model

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

def collect_hiddens(data, label, MODEL_PATH, tag):
    for modelp in os.listdir(MODEL_PATH):
        if '.pt' in modelp and '.npy' not in modelp:
                for idx, seq in enumerate(data):
                    h_steps = []
                    for tstep in range(1, seq.shape[0]+1):
                        model = torch.load(MODEL_PATH+modelp, map_location='cpu')
                        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                        model.to(device)
                        model.eval()
                        optimizer.zero_grad()
                        h1 = model.init_hidden1(1, 1)
                        ins, X_length = load_batch(seq[:tstep, :])
                        ins = ins.to(device)
                        outs, hid = model(ins, X_length, h1)
                        _, preds = outs.max(1)
                        h_steps.append(np.abs(hid[0].detach().cpu().numpy().squeeze()).astype('str'))
                    h_steps = np.insert(h_steps, label, tag)
                    if preds.item() == label:
                        swap_acc.append(idx)
                    sh_all.append(h_steps)
            np.save((MODEL_PATH+tag +'/'+modelp.split(']')[0]+'-'+tag+'-all'+np.asarray(sh_all))
            np.save((MODEL_PATH+tag +'/'+modelp.split(']')[0]+'-'+tag+'-acc-'+str(len(swap_acc))), np.asarray(swap_acc))
