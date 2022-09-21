import numpy as np
import pandas as pd
import faiss
from refile import smart_open    
import nori2 as nori
import io
from PIL import Image
import matplotlib.pyplot as plt
from utils.plot_pairs import plot_pairs

feature_file = 'cache/resnet18-random-yfcc14m.npy'

nori_fetcher = nori.Fetcher()
def get_img_from_nori_id(nori_id):
    return Image.open(io.BytesIO(nori_fetcher.get(nori_id)))

print(f'loading {feature_file}...')
features = np.load(feature_file).astype('float32')
d = features.shape[1]
print(features.shape)

index = faiss.IndexFlatL2(d)
index.add(features)

k = 5
n_query = 1000000
D, I = index.search(features[:n_query], k)
# print(I)
# print(D)

dataset_csv = 'cache/yfcc_nori.csv'
if dataset_csv[:2]=='s3':
    df = pd.read_csv(smart_open(dataset_csv, "r"), sep='\t')
else:
    df = pd.read_csv(dataset_csv, sep='\t')
print(df)

captions = df['title'].tolist()
nori_ids = df['filepath'].tolist()

images = []
texts = []
labels = []
dists = []

for i in range(len(I)):
    results = I[i]
    for r in range(len(results)):
        #images.append(get_img_from_nori_id(nori_ids[int(I[i][r])]))
        images.append(nori_ids[int(I[i][r])])
        texts.append(str(captions[int(I[i][r])]).replace('\t', ' '))
        labels.append(i)
        dists.append(D[i][r])

knn_df = pd.DataFrame({
    'images':images,
    'labels':labels,
    'dists':dists,
    'texts':texts,
})
print(knn_df)
knn_df.to_csv(f"k={k}_n_query={n_query}_feature={feature_file.replace('cache/', '')}.csv", index=False, sep='\t')


# plt.hist(NN_dist, bins=100)
# plt.savefig(f"NN_dist_hist-k={k}_feature={feature_file.replace('cache/', '')}.png")

# plot_pairs(images, texts, sample_per_row=k, file_name=f"k={k}_feature={feature_file.replace('cache/', '')}.png")