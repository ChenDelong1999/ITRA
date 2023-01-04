
import torch
import torch.nn as nn

import numpy as np
from sklearn.linear_model import LogisticRegression as sklearnLogisticRegression
from torch.utils.data import DataLoader
from data.classification_datasets import get_dataset
from tqdm import tqdm
import logging

def logistic_regression_pytorch(train_features, train_labels, test_features, test_labels, total_epochs=100, lr=0.004, weight_decay=0, batch_size=1024):
    
    class AverageMeter(object):
        """computes and stores the average and current value"""

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    class TensorDataset():
        def __init__(self, *tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return tuple(tensor[index] for tensor in self.tensors)

        def __len__(self):
            return self.tensors[0].size(0)
        
    class Classifier(nn.Module):
        def __init__(self, feature_dim, num_labels):
            super(Classifier, self).__init__()

            self.linear = nn.Linear(feature_dim, num_labels)
            self.linear.weight.data.normal_(mean=0.0, std=0.01)
            self.linear.bias.data.zero_()

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.linear(x)

    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    train_dataset = TensorDataset(torch.Tensor(train_features), torch.Tensor(train_labels).long())
    val_dataset = TensorDataset(torch.Tensor(test_features), torch.Tensor(test_labels).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=5000, num_workers=4, pin_memory=True, persistent_workers=True)
    
    
    num_labels = int(max(train_labels)+1)
    classifier = Classifier(train_features.shape[1], num_labels).cuda()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs, eta_min=0)
    criterion = nn.CrossEntropyLoss().cuda()
    best_acc = 0
    for epoch in (pbar := tqdm(range(total_epochs))):
        top1_train = AverageMeter()
        top1 = AverageMeter()
        losses = AverageMeter()

        for step, (feature, label) in enumerate(train_loader):
            feature = feature.cuda()
            label = label.cuda()
            output = classifier(feature)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1 = accuracy(output, label, topk=(1, ))[0]
            losses.update(loss.item(), feature.size(0))
            top1_train.update(acc1[0], feature.size(0))
        
        for step, (feature, label) in enumerate(val_loader):
            feature = feature.cuda()
            label = label.cuda()
            with torch.no_grad():
                output = classifier(feature)
            acc1 = accuracy(output, label, topk=(1, ))[0]
            top1.update(acc1[0], feature.size(0))

        scheduler.step()
        
        if top1.avg.item() > best_acc:
            best_acc = top1.avg.item()
        pbar.set_description(f'Epoch {epoch+1}, test accuracy {top1.avg.item():.2f}, best accuracy {best_acc:.2f}')

    return best_acc
        
@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    train_features = torch.from_numpy(train_features).cuda()
    train_labels = torch.from_numpy(train_labels).cuda()
    test_features = torch.from_numpy(test_features).cuda()
    test_labels = torch.from_numpy(test_labels).cuda()
    train_features = nn.functional.normalize(train_features, dim=1, p=2)
    test_features = nn.functional.normalize(test_features, dim=1, p=2)

    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighborsresnet18_distill_resnet50-moco-v2-checkpoint_0199.pth.tar
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        #top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    #top5 = top5 * 100.0 / total
    return top1#, top5


def get_features(model, dataset, args):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)):
            images = images.to(args.device)

            if args.distributed and not args.horovod:
                image_features = model.module.encode_image(images)
            else:
                image_features = model.encode_image(images)

            all_features.append(image_features.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


def get_dataset_features(model, dataset_name, root, preprocess, args):

    # if dataset_name=='cifar10':
    #     train = CIFAR10(root, download=True, train=True, transform=preprocess)
    #     test = CIFAR10(root, download=True, train=False, transform=preprocess)

    # elif dataset_name=='cifar100':
    #     train = CIFAR100(root, download=True, train=True, transform=preprocess)
    #     test = CIFAR100(root, download=True, train=False, transform=preprocess)

    # elif dataset_name=='stl10':
    #     train = STL10(root, download=True, split='train', transform=preprocess)
    #     test = STL10(root, download=True, split='test', transform=preprocess)
    # # TODO: remove nori dependency
    # elif dataset_name=='imagenet':
    #     train = ImageNet_nori(split='train', transform=preprocess)
    #     test = ImageNet_nori(split='val', transform=preprocess)
    # elif dataset_name=='imagenet-50k':
    #     train = ImageNet_50k(transform=preprocess)
    #     test = ImageNet_nori(split='val', transform=preprocess)
    # else: 
    #     train = ImageFolder(f'{args.datasets_dir}/{dataset_name}/train', transform=preprocess)
    #     test = ImageFolder(f'{args.datasets_dir}/{dataset_name}/test', transform=preprocess)

    train = get_dataset(dataset_name=dataset_name, split='train', root=args.datasets_dir, transform=preprocess)
    test = get_dataset(dataset_name=dataset_name, split='test', root=args.datasets_dir, transform=preprocess)

        
    # Calculate the image features
    logging.info(f'extracting featres from {dataset_name} training set...')
    train_features, train_labels = get_features(model, train, args=args)
    logging.info(f'extracting featres from {dataset_name} testing set...')
    test_features, test_labels = get_features(model, test, args=args)
    return train_features, train_labels, test_features, test_labels


def get_linear_eval_acc(train_features, train_labels, test_features, test_labels, args):
    
    if args.linear_prob_mode=='sklearn':
        logging.info('Runing sklearn-based logistic regression')
        classifier = sklearnLogisticRegression(random_state=0, C=args.C, max_iter=1000, verbose=1, n_jobs=-1)
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(test_features)
        accuracy = 100 * np.mean((test_labels == predictions).astype(np.float)) 

    elif args.linear_prob_mode=='sklearn-search':
        logging.info('Runing sklearn-based logistic regression with hyper-parameter search...')
        accuracies = []
        Cs = [1e-2, 1e-1, 1, 2, 4, 8, 10, 16, 64 ,1e2, 1e3]
        for C in Cs:
            logging.info(f'Runing sklearn-based logistic regression with C={C}')
            classifier = sklearnLogisticRegression(random_state=0, C=C, max_iter=50, verbose=0, n_jobs=-1)
            classifier.fit(train_features, train_labels)
            predictions = classifier.predict(test_features)
            accuracy = 100 * np.mean((test_labels == predictions).astype(np.float)) 
            accuracies.append(accuracy)
            logging.info(f'accuracy={accuracy}')
        accuracy = max(accuracies)
        index = accuracies.index(accuracy)
        logging.info(f'Get all accuracies: {str(accuracies)}, the best one {accuracy} is achieved by C={Cs[index]}')

        classifier = sklearnLogisticRegression(random_state=0, C=Cs[index], max_iter=1000, verbose=1, n_jobs=-1)
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(test_features)
        accuracy = 100 * np.mean((test_labels == predictions).astype(np.float)) 
    
    elif args.linear_prob_mode=='pytorch' or 'ImageNet':
        logging.info('Runing pytorch-based logistic regression')
        accuracy = logistic_regression_pytorch(
            train_features, train_labels, test_features, test_labels, 
            #total_epochs=500, lr=0.2, weight_decay=1e-5, batch_size=10000
            )
        
    elif args.linear_prob_mode=='pytorch-search':
        logging.info('Runing pytorch-based logistic regression with hyper-parameter search...')
        accuracies = []
        for _ in range(5):
            lr = 10 ** (-np.random.randint(low=1, high=3)) * np.random.randint(low=1, high=9)
            weight_decay = 10 ** (-np.random.randint(low=5, high=10)) * np.random.randint(low=1, high=9)
            logging.info(f'Runing pytorch-based logistic regression with random hyper-parameters: lr={lr}, wd={weight_decay}')
            accuracy = logistic_regression_pytorch(train_features, train_labels, test_features, test_labels, total_epochs=500, lr=lr, weight_decay=weight_decay, batch_size=10000)
            logging.info(f'Accuracy={accuracy}')
            accuracies.append(accuracy)
        accuracy = max(accuracies)
        logging.info(f'Got all accuracies: {str(accuracies)}, best={accuracy}')
    return  float(accuracy)


def get_knn_acc(train_features, train_labels, test_features, test_labels, args):
    top1 = knn_classifier(train_features, train_labels, test_features, test_labels, 20, 0.07, num_classes=int(max(train_labels)+1))
    return  float(top1)


SKLEARN_LINEAR_C = {
    'MNIST':0.1, 
    'CIFAR10':2, 
    'CIFAR100':0.1, 
    'STL10':0.01, 
    'StanfordCars':0.1, 
    'DTD':0.1, 
    'Food101':0.1, 
    'OxfordIIITPet':0.1, 
    'RenderedSST2':0.1,
    'ImageNet':8, 
    'ImageNet-50k':8
}



def linear_eval(model, dataset_names, epoch, preprocess, args):
      
    if args.linear_frequency == 0:
        return {}
    if (epoch % args.linear_frequency) != 0 and epoch != args.epochs:
        return {}

    results = {}
    for dataset_name in dataset_names:
        logging.info(f'starting linear evaluation on {dataset_name}...')
        train_features, train_labels, test_features, test_labels = get_dataset_features(model, dataset_name, args.datasets_dir, preprocess, args)
        
        if args.linear_prob_mode=='ImageNet':
            knn_acc = get_knn_acc(train_features, train_labels, test_features, test_labels, args)
            results[f'{dataset_name}-knn-eval-acc'] = knn_acc
            logging.info(f'Finished K-NN evaluation on  {dataset_name}, accuracy: {knn_acc}')
            return results

        if dataset_name in SKLEARN_LINEAR_C.keys():
            args.C = SKLEARN_LINEAR_C[dataset_name]
        linear_acc = get_linear_eval_acc(train_features, train_labels, test_features, test_labels, args)
        results[f'{dataset_name}-linear-eval-acc'] = linear_acc
        logging.info(f'Finished linear evaluation on  {dataset_name} accuracy: {linear_acc}')
        
        if dataset_name != 'ImageNet':
            knn_acc = get_knn_acc(train_features, train_labels, test_features, test_labels, args)
            results[f'{dataset_name}-knn-eval-acc'] = knn_acc
            logging.info(f'Finished K-NN evaluation on  {dataset_name}, accuracy: {knn_acc}')

    return results



if __name__=='__main__':
    import open_clip
    import os
    import pickle as pkl
    model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='OpenAI')
    train_data, val_data = pkl.load(open(os.path.join('train.pkl'),'rb')), pkl.load(open(os.path.join('test.pkl'),'rb'))
    train_features, train_labels = train_data['features'], train_data['labels']
    test_features, test_labels = val_data['features'], val_data['labels']
    logistic_regression_pytorch(train_features, train_labels, test_features, test_labels)
