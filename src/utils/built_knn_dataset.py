import numpy as np
import pandas as pd
import faiss
from refile import smart_open    
import nori2 as nori
import io
from PIL import Image
import matplotlib.pyplot as plt
from utils.plot_pairs import plot_pairs


k=10
dataset_random_csv = 'k=10_n_query=100000_feature=resnet18-random-yfcc14m.npy.csv'
dataset_pretrained_csv = 'k=10_n_query=100000_feature=resnet18-yfcc14m.npy.csv'
df_random = pd.read_csv(dataset_random_csv, sep='\t')
df_pretrained = pd.read_csv(dataset_pretrained_csv, sep='\t')
print(df_random)
print(df_pretrained)

labels = np.array(df_random['labels'].tolist())
dists_random = np.array(df_random['dists'].tolist())
texts_random = np.array(df_random['texts'].tolist())
dists_pretrained = np.array(df_pretrained['dists'].tolist())
texts_pretrained = np.array(df_pretrained['texts'].tolist())

all_index = np.arange(len(labels))
nn_mask = all_index[all_index % k == 0] + 1
knn_mask = all_index[all_index % k != 0]
print(nn_mask, len(nn_mask))
print(knn_mask, len(knn_mask))

plt.figure(figsize=[20,20])
plt.scatter(
    x=dists_random[knn_mask],
    y=dists_pretrained[knn_mask],
    s=0.2,
    alpha=0.8,)
plt.xlim([-1, 30])
plt.ylim([-20, 600])
plt.savefig('knn_mask.png')


plt.figure(figsize=[20,20])
plt.scatter(
    x=dists_random[nn_mask],
    y=dists_pretrained[nn_mask],
    s=0.2,
    alpha=0.8,)
plt.xlim([-1, 30])
plt.ylim([-20, 600])
plt.savefig('nn_mask.png')