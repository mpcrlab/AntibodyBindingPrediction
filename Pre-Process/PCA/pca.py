
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)



MODEL_PATH = 'results/top_models/LSTM_small/knockout_peaks' + '/'


files = os.listdir(MODEL_PATH)
files.sort()
bind = []
nob = []
blen = 0
nlen = 0
for file in files:
    if '.npy' in file :
        if 'bind_acc' in file and 'F' not in file:
            b = np.load(MODEL_PATH+file)
            print(file)
            blen += len(b)
        if 'no_acc' in file and 'F' not in file:
            n = np.load(MODEL_PATH+file)
            print(file)
            nlen += len(n)
        if 'h_all' in file and 'F' not in file:
            h = np.load(MODEL_PATH+file)
            bind.append(np.asarray(h[:81])[b])
            nob.append(np.asarray(h[81:])[n])
            print(file)


all = []
shapes = []
for b in bind:
    for s in b:
        all.append(s[1:].astype(float))
        shapes.append(len(s))

for b in nob:
    for s in b:
        all.append(s[1:].astype(float))
        shapes.append(len(s))

m = max(shapes)


a = []
for s in all:
    p = np.zeros(m-len(s))
    a.append(np.append(s,p))

all = 0

df = pd.DataFrame(np.asarray(a))
df = df.fillna(0)


color = np.append(np.ones((blen)), np.zeros((nlen)), axis=0)
df.insert(loc=0, column='color', value=color)


##2d Plot
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[df.keys()[0:]].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
plt.close()
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue='color',
    palette=sns.color_palette("bright", 2),
    data=df,
    legend="full",
    alpha=0.3
)
plt.title('2d PCA Hidden States per Class')
plt.show()


##3d Plot
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[df.keys()[1:]].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
plt.close()
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[:,"pca-one"],
    ys=df.loc[:, "pca-two"],
    zs=df.loc[:, "pca-three"],
    c=df['color'],
    cmap='Spectral'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.title('3d PCA Hidden States per Class')
plt.show()
