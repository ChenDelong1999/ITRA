"""
https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/kmeans.py

This is a simple application for sentence embeddings: clustering
Sentences are mapped to sentence embeddings and then k-mean clustering is applied.
"""
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd
# TODO: remove nori dependency when publish codes
from refile import smart_open
import nori2 as nori
import io
from PIL import Image
import torch
import numpy as np
import faiss
from utils.plot_pairs import plot_pairs


class CsvDataset():
    def __init__(self, input_filename='s3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv', img_key='filepath', caption_key='title', sep="\t"):
        if input_filename[:2]=='s3':
            self.using_nori = True
            df = pd.read_csv(smart_open(input_filename, "r"), sep=sep)
            self.f = None
        else:
            self.using_nori = False
            df = pd.read_csv(input_filename, sep=sep)
        print(df)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()

        import re
        for i in range(len(self.captions)):
            self.captions[i] = ' '.join(re.split(' |.', self.captions[i])[:77])
        
    def get_data(self, idx):
            
        # get image data
        if self.using_nori:
            if self.f is None:
                self.f = nori.Fetcher()
            pic = Image.open(io.BytesIO(self.f.get(self.images[idx])))
        else:
            pic = Image.open(str(self.images[idx]))
        
        texts = self.captions[idx]
        
        return pic, texts

def kmeans(feature, k):
    centroids = torch.zeros([k, feature.shape[1]])    
    kmeans = faiss.Kmeans(
        d=feature.shape[1], 
        k=k, 
        niter=50, 
        nredo=1,
        verbose=True, 
        gpu=True)
    kmeans.train(feature)

    # in case of derived centroid is less than args.k
    centroids[:,:kmeans.centroids.shape[1]] = torch.from_numpy(kmeans.centroids)
    distance, labels = kmeans.index.search(feature, 1)
    labels = np.array(labels)
    labels = np.reshape(labels, labels.shape[0])

    return torch.from_numpy(labels)

def show_samples(dataset, label, file_name, sample_per_class=16, max_rows=16):
    images = []
    texts = []    
    unique_labels, n_samples = np.unique(label, return_counts=True)

    for k in unique_labels[:max_rows]:
        cluster_dataset_index = np.squeeze(np.argwhere(label==k))
        if cluster_dataset_index.shape==():
            continue # empty cluster
        # show [sample_per_class] samples for each class
        for i in range(sample_per_class): 
            # sometimes there are not much sample in this cluster
            if i >= len(cluster_dataset_index):
                images.append(Image.new('RGB', (256,256), (255,255,255)))
                texts.append(' ')
            else:
                image, text = dataset.get_data(int(cluster_dataset_index[i]))
                images.append(image)
                texts.append(text)

    # TODO: this is a nori dependency
    dataset.f = None
                                
    plot_pairs(
        images[:100*sample_per_class], texts[:100*sample_per_class], 
        suptitle=file_name, file_name=file_name+'.png', 
        sample_per_row=sample_per_class
    )  

if __name__=='__main__':
    dataset = CsvDataset()
    #for model in ['average_word_embeddings_glove.6B.300d', 'all-MiniLM-L6-v2', 'all-mpnet-base-v2']:
    for model in ['clip-ViT-L-14']:
        embedder = SentenceTransformer(model)

        # Corpus with example sentences
        corpus = dataset.captions[:2500000]

        corpus_embeddings = embedder.encode(corpus, show_progress_bar=True, batch_size=512)

        # Perform kmean clustering
        num_clusters = 10000
        cluster_assignment = kmeans(corpus_embeddings, num_clusters)

        #clustering_model = KMeans(n_clusters=num_clusters)
        #clustering_model.fit(corpus_embeddings)
        #cluster_assignment = clustering_model.labels_

        clustered_sentences = [[] for i in range(num_clusters)]
        clustered_ids = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(corpus[sentence_id])
            clustered_ids[cluster_id].append(sentence_id)

        show_samples(dataset, cluster_assignment, f'vis_cluster_[{model}]', sample_per_class=10, max_rows=64)
    
    # for i, cluster in enumerate(clustered_sentences):
    #     print("Cluster ", i+1)
    #     print(cluster)
    #     print("")
