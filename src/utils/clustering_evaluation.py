
import numpy as np
import pandas as pd
import faiss
import time

# https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def clustering_evaluation(features, k):

    features = features.astype(np.float32)
    kmeans = faiss.Kmeans(
            d=features.shape[1], 
            k=k, 
            niter=50, 
            nredo=1,
            verbose=False, 
            gpu=True)
    kmeans.train(features)

    distance, label = kmeans.index.search(features, 1)
    label = np.array(label)
    label = np.reshape(label, label.shape[0])

    end = time.time()
    SIL_score = silhouette_score(features, label, sample_size=min(len(features), 10000))
    print('SIL_score time:', time.time()-end)
    
    end = time.time()
    DBI_score = davies_bouldin_score(features, label)
    print('DBI_score time:', time.time()-end)
    
    end = time.time()
    CHI_score = calinski_harabasz_score(features, label)
    print('CHI_score time:', time.time()-end)

    return SIL_score, DBI_score, CHI_score


def loop(features):
    SIL_scores, DBI_scores, CHI_scores = [], [], []
    num_clusters = np.arange(start=10, stop=int(len(features)), step=int(len(features)/10))
    print('features:', features.shape)
    print('num_clusters:', num_clusters)

    for k in num_clusters:
        SIL_score, DBI_score, CHI_score = clustering_evaluation(features, int(k))
        SIL_scores.append(SIL_score)
        DBI_scores.append(DBI_score)
        CHI_scores.append(CHI_score)
        print(f'k={k},\t{SIL_score}, \t{DBI_score}, \t{CHI_score}')
    df = pd.DataFrame.from_dict(data={
        'k':num_clusters,
        'SIL_scores':SIL_scores,
        'DBI_scores':DBI_scores,
        'CHI_scores':CHI_scores,
    })
    print(df)


all_image_features, all_text_features = np.load('cache/features.npy')
print('all_text_features')
loop(all_text_features)
