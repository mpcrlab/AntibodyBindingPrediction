
import os, sys
import numpy as np
from numpy import cov
import matplotlib.pyplot as plt
from scipy.stats import pearsonr



HIDDENS_PATH = '' + '/'
SAVE_PATH = '' + '/'

##############################


np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


files = os.listdir(HIDDENS_PATH)
files.sort()

def load_hiddens(tag, HIDDENS_PATH):
    container = []
    for file in files:
        if '.npy' in file :
            if tag in file and 'acc' in file:
                s = np.load(MODEL_PATH+file)
                print(file)
            if tag in file and 'all' in file:
                print(file)
                container.append(np.asarray(np.load(MODEL_PATH+file))[s])
    return container

def pad(bin):
    shapes = []
    for b in [np.asarray([i]).astype(float).squeeze() for i in bin]:
        shapes.append(b.shape[0])
    M = max(shapes)
    padbin = []
    for b in bin:
        ad = np.zeros([M-b.shape[0], 300])
        c = np.append(b, ad, axis=0)
        padbin.append(c.astype(float))
    avgs = np.average(padbin, axis=0)
    return shapes, padbin, avgs

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
    bb = []
    for b in bin:
        for s in b:
            bb.append(np.flip(s[1:].reshape(s.shape[0]//300, 300) ,0))
    bclip = []
    for b in bb:
        bclip.append(b[:400,].astype(float))
    shapes, bpadbin, bavgs = pad(bclip)
    bsum = np.sum(np.asarray(bpadbin), axis=0)
    bss = np.sum(bsum, axis=1)
    b = f(bss)
    return b

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
