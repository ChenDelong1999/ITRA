import torch
import numpy as  np



def mean_classifier_acc(features, labels):
    """
    feautres: (nxd)
    lables: (n, )
    """
    n_feature = features.shape[1]
    n_class = len(np.unique(labels))
    classifier = np.zeros(shape=(n_class, n_feature))
    for i in range(n_class):
        c = np.unique(labels)[i]
        mean_feature = features[labels==c].mean(axis=0)
        classifier[i] = mean_feature
    
    prediction = (features @ classifier.T).argmax(axis=1)
    accuracy = 100 * np.mean((labels == prediction).astype(float)) 

    # print('features')
    # print(features)
    # print('labels')
    # print(labels)
    # print('classifier')
    # print(classifier)
    # print('prediction')
    # print(prediction)

    return accuracy
    
    
if __name__=='__main__':
    n_class = 2
    n_feature = 128
    n_sample = 10000
    all_acc = []
    for i in range(1):
        features = np.random.randn(n_sample, n_feature)
        labels = np.array(np.arange(n_class).tolist() * int(n_sample/n_class))
        print(features.shape, labels.shape)
        acc = mean_classifier_acc(features, labels)
        all_acc.append(acc)
    print(all_acc)
    print(np.mean(all_acc))


