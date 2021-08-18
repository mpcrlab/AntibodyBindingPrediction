print('Hello...')
import csv
import time
import copy
import random
import os, sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset





import wandb
wandb.init(project='antibodies-flipped-CNN_sig')
wab = wandb.config



parser = argparse.ArgumentParser()
parser.add_argument('Train', help='True or False')
parser.add_argument('Data', help='Please input the location of your data files here.')
parser.add_argument('N_layers', help='Please input the number of LSTM layers')
parser.add_argument('N_hidden', help='Please input the number of hidden units in each LSTM layer')
parser.add_argument('Optimizer', help='adam or sgd, default=adam')
parser.add_argument('Dropout', help='amount of dropout, value between 0 and 1, default=0.5')
parser.add_argument('Learning_rate', help='learning_rate value, default=0.001')
parser.add_argument('Epochs', help='number of epochs, default=200')
parser.add_argument('Save', help='path of folder where to save trained models')
parser.add_argument('Test', help= 'True or False')
parser.add_argument('Load', help='Path of saved model to test.')
parser.add_argument('Analysis',help='Analyze hidden states, True or False.')
parser.add_argument('Hiddens', help='Path where to save hidden states.')
parser.add_argument('Results', help='Path where ot save the results of analysis.')
args=parser.parse_Args()


df = args.Data


wab.NUM_CLASSES = 2
wab.N_LAYERS = args.N_layers else 1
wab.INPUT_DIM = 27
wab.HIDDEN_DIM = args.N_hidden else 200
wab.DROPOUT = args.Dropout else .5
wab.LR = args.Learning_rate else 0.001
wab.BS = 1
wab.NUM_EPOCHS = args.Epochs else 200
wab.OPTIM = str(args.Optimizer) else 'adam'

PRESAVE_NAME = str(arg.Save)


#some stuff for wandb
nl = wab.N_LAYERS
hd = wab.HIDDEN_DIM
ba = wab.BS


"""load the data"""
if Train:
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
	print('Using device: ', device)



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


	model = Classifier_LSTM(nl, hd,ba)
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
	    print('Best Val Acc: {:4f}'.format(epoch_acc.detach().cpu().item()))
	    return model, best_acc


	model, best = train()




###########################################################################
if args.Test:
	if Train and if Test:
		MODEL_LOAD_DIR = SAVE_NAME
	else: 
		MODEL_LOAD_DIR = str(args.Load) + '/'
	NUM_CLASSES = 2
	INPUT_DIM = 27
	DROPOUT = args.Dropout else .5
	LR = args.Learning_rate else 0.001
	HD = args.N_hidden else 200
	NL = args.N_layers else 1
	BS = 1

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

	if Train == False:
		np_load_old = np.load
		np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
	xtrain = np.load(df+'/bind/bind_train.npy')
	ntrain = np.load(df+'/notbind/nobind_train.npy')
	xtest = np.load(df+'/bind/bind_test.npy')
	ntest = np.load(df+'/notbind/nobind_test.npy')
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

	print('Testing bind:')
	test(bhot_seqs, 1)
	print('Testing not-bind:')
	test(nhot_seqs, 0)



##########################################################################

if args.Analysis:
	NUM_CLASSES = 2
	INPUT_DIM = 27
	DROPOUT = args.Dropout else .5
	LR = args.Learning_rate else 0.001
	HD = args.N_hidden else 200
	NL = args.N_layers else 1
	BS = 1

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
		if Train and if test == False:
			np_load_old = np.load
			np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
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

	collect_hiddens(bhot_seqs, 1, args.Hiddens, 'bind')
	collect_hiddens(nhot_seqs, 0, args.Hiddens, 'notbind')


	HIDDENS_PATH = args.Hiddens + '/'
	SAVE_PATH = args.Results + '/'

	files = os.listdir(HIDDENS_PATH)
	files.sort()

	def load_hiddens(tag, HIDDENS_PATH):
		c = []
	    for file in files:
	        if '.npy' in file :
	            if tag in file and 'acc' in file:
	                s = np.load(MODEL_PATH+file)
	                print(file)
	            if tag in file and 'all' in file:
	                print(file)
	                c.append(np.asarray(np.load(MODEL_PATH+file))[s])
	    return c

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

	def stats(b,n):
	    i = 6
	    t = 400
	    num = .58
	    a = smooth(np.abs(normies(b)-normies(n)), i)
	    # ab = smooth(np.abs(normies(d)-normies(n)), i)
	    # x = thresh(ab-a)
	    x = thresh(a)
	    x = np.where(x>np.max(x)*(num), x, 0)
	    return x

	def peak_locations(x):
	    w = []
	    for i,s in enumerate(x):
	        if s !=0:
	            w.append(i)
	    print(len(w))
	    np.save(SAVE_PATH+str(num), np.asarray(w))



	cb = load_hiddens('bind')
	cn = load_hiddens('notbind')
	b = process(cb)
	n = process(cn)
	x = stats(b,n)
	peak_locations(x)
