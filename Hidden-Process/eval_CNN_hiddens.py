
import csv
import time
import copy
import random
import os, sys
import numpy as np
from numpy import cov
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

FLIPPED = True
INSERT = False
KNOCKOUT = False
KNOCKOUT_PEAKS = True

LOAD_PATH = ''
BS = 1
INPUT_DIM = 27




def get_data():
    xtrain = np.load('data/bind/bind_train.npy')
    ntrain = np.load('data/notbind/nobind_train.npy')
    xtest = np.load('data/bind/bind_test.npy')
    ntest = np.load('data/notbind/nobind_test.npy')
    sxtrain = xtrain.copy()
    sxtest = xtest.copy()
    sntrain = ntrain.copy()
    sntest = ntest.copy()
    if flipped == True:
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

def hot_prots(X):
    X_bin = []
    ide = np.eye(INPUT_DIM, INPUT_DIM)
    for i in range(X.shape[0]):
        x_ = X[i]
        x = ide[x_.astype(int),:]
        X_bin.append(x)
    return X_bin

def cnn_batch(x,y):
    longest = 1750
    ad = np.zeros((longest-x.shape[0], INPUT_DIM))
    ad[:, 26] += 1
    im = np.concatenate((x, ad), axis=0)
    labels = torch.from_numpy(y).to(torch.long)
    ins = torch.from_numpy(np.asarray(im)).to(torch.float)
    return ins, labels



# target seq and tests

dd = ['QVKLLESGPELVKPGASVKMSCKASGYTFTSYVMHWVKQKPGQGLEWIGYINPYNDGTKYNEKFKGKATLTSDKSSSTAYMELSSLTSEDSAVYYCVRGGYRPYYAMDYWGQGTSVTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPRDCTSHHHHHHELQMTQSPASLSASVGETVTITCRASENIYSYLAWYQQKQGKSPQLLVYNAKTLAEGVPSRFSGSGSGTQFSLKINSLQPEDFGSYYCQHHYGTPLTFGAGTKLELKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNEC']
ds = []
for c in dd[0]:
    ds.append(int(ord(c.lower())-97))

ds = np.expand_dims(ds, 0)
d = np.asarray([np.flip(i) for i in ds])
DX = np.asarray(hot_prots(d))[0]
DX = DX[None, :,:]


#model and feature maps
class ConvNet(nn.Module):
    def __init__(self, classes = 2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            nn.Dropout(.5),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 102, kernel_size=5, stride=1, padding=1),
            nn.Dropout(.5),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(102, 1, kernel_size=1, stride=1, padding=1),
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook


def get_maps(x, y):
    container = []
    for m in os.listdir(LOAD_PATH):
        if '.pt' in m:
            model = torch.load(LOAD_PATH + '/' +m)
            model= model.to(device)
            model.eval()
            all = []
            for seq in x:
                s, y = cnn_batch(seq, np.asarray([y]))
                s = s[None, None, :,:].to(device)
                activation = {}
                model.layer3.register_forward_hook(get_activation('layer3'))
                predictions = F.softmax(model(s)).detach().cpu().numpy()
                if np.argmax(predictions) == y:
                    act = activation['layer3'].squeeze()
                    free_act = act.detach().cpu().numpy()
                    all.append(free_act)
            container.append(all)
    return container


#########################

dd = ['QVKLLESGPELVKPGASVKMSCKASGYTFTSYVMHWVKQKPGQGLEWIGYINPYNDGTKYNEKFKGKATLTSDKSSSTAYMELSSLTSEDSAVYYCVRGGYRPYYAMDYWGQGTSVTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPRDCTSHHHHHHELQMTQSPASLSASVGETVTITCRASENIYSYLAWYQQKQGKSPQLLVYNAKTLAEGVPSRFSGSGSGTQFSLKINSLQPEDFGSYYCQHHYGTPLTFGAGTKLELKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNEC']
ds = []
for c in dd[0]:
    ds.append(int(ord(c.lower())-97))

ds = np.expand_dims(ds, 0)

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


X =  hot_prots(np.asarray(X))
print('# of bind test sequences:', len(X))

################################################################################


def shapes(bin):
    shapes = []
    for b in [np.asarray([i]).astype(float).squeeze() for i in bin]:
        shapes.append(b.shape[0])
    return shapes

def thresh(bin):
    for i,a in enumerate(bin):
        if a <=0:
            bin[i,] = 0
    return bin

def f(x):
    n = (x-np.min(x))
    den = np.max(x)-np.min(x)
    return (n/den)

def smooth(bin, k):
    return np.convolve(bin, np.ones(k)/k, 'same')

def makey(m):
        Y = np.zeros(400)
        Y[30:35] = m
        Y[49:66] = m
        Y[98:109] = m
        Y[253:264] = m
        Y[278:286] = m
        Y[318:327] = m
        return Y

def normies(bin):
    mean = np.mean(np.abs(bin))
    std = np.std(bin)
    norm = bin-mean
    norm = norm/std
    return norm

def process(bin):
    b = np.asarray(bin)
    bb = np.sum(b, 0)
    bbb = np.sum(bb,0)
    bbbb = np.sum(bbb, 1)
    return bbbb[:400]


def stats(test, target):
    i = 6
    t = 400
    num = .58
    a = smooth(np.abs(normies(s)-normies(n)), i)
    ab = smooth(np.abs(normies(d)-normies(n)), i)
    x = thresh(ab-a)
    x = np.where(x>np.max(x)*(num), x, 0)
    c = cov(x,Y)
    corr, _ = pearsonr(x,Y)
    print(c, '\n', '\n', corr, '\n')

def peak_locations(x):
    w = []
    for i,s in enumerate(x):
        if s !=0:
            w.append(i)
    print(len(w))
    np.save(SAVE_PATH+str(num), np.asarray(w))

def peak_letters(x, target_seq):
    xline = list(np.tile(np.asarray(['']), (len(x))))
    for i in w:
        xline[i] = dd[0][i]
    print(xline)


def overlap(test, target):
    i = 6
    t = 400
    num = .39
    a = smooth(np.abs(normies(s)-normies(n)), i)
    ab = smooth(np.abs(normies(d)-normies(n)), i)
    x=thresh(ab-a)
    x = np.where(x>np.max(x)*(num), x, 0)
    xd = np.where(ab>np.max(ab)*(num), ab, 0)
    both = []
    for i, num in enumerate(x):
        if num >0 and xd[i] >0:
            both.append(i)
    xsubs = np.zeros([len(x)])
    dsubs = np.zeros([len(xd)])
    for i, num in enumerate(x):
        if i in both:
            xsubs[i] = x[i]
            dsubs[i] = xd[i]











#
