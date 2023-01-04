
import logging
import tqdm
import numpy as np
import faiss

import torch
import torch.nn.functional as F
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, STL10
# TODO: disable nori

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from data.train_data import ImageNet_nori
from data.classification_datasets import get_dataset
from sklearn.metrics import confusion_matrix

def zero_shot_classifier(model, classnames, templates, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.replace('{}',classname) for template in templates]    
            
            if args.distributed and not args.horovod:
                class_embeddings = model.module.encode_text(texts, projection=True)
            else:
                class_embeddings = model.encode_text(texts, projection=True)
            # class_embeddings = teacher.encode(
            #     texts,
            #     convert_to_tensor=True, 
            #     show_progress_bar=False
            #     )
            class_embeddings = class_embeddings.mean(dim=0)
            class_embedding = F.normalize(class_embeddings, dim=-1)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding.cpu())
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)#.to(args.device)
    return zeroshot_weights


def _accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def mean_class_accuracy(scores, labels):
    pred = np.argmax(scores, axis=1)
    cf = confusion_matrix(labels, pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    return np.mean(cls_hit/cls_cnt) * 100

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    return float(correct[0].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) * 100 / len(target)

def run(model, classifier, dataloader, args, mean_per_class):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        acc_mean_per_class = 0.
        all_image_features = []
        all_labels = []
        all_logits = []
        for images, target in tqdm.tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device)

            if args.distributed and not args.horovod:
                image_features = model.module.encode_image(images, projection=True)
            else:
                image_features = model.encode_image(images, projection=True)

            image_features = F.normalize(image_features, dim=-1).detach().cpu()
            logits = 100. * image_features @ classifier

            all_image_features.append(image_features)
            all_labels.append(target)
            all_logits.append(logits)

            # acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            # acc1 = accuracy(logits, target, topk=(1,))[0]
            # top1 += acc1
            # n += images.size(0)

    all_image_features = torch.cat(all_image_features)
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)

    if mean_per_class:    
        logging.info('Using Mean-per-class Accuracy')
        acc = mean_class_accuracy(all_logits, all_labels)
    else:
        acc = accuracy(all_logits, all_labels, topk=(1,))
    return round(acc, 2), all_image_features, all_labels

def clustering_evaluation(features, labels):

    features = features.numpy().astype(np.float32)
    kmeans = faiss.Kmeans(
            d=features.shape[1], 
            k=int(max(labels)+1), 
            niter=100, 
            nredo=5,
            verbose=False, 
            gpu=True)
    kmeans.train(features)

    distance, img_plabels = kmeans.index.search(features, 1)
    img_plabels = np.array(img_plabels)
    img_plabels = np.reshape(img_plabels, img_plabels.shape[0])

    ari = adjusted_rand_score(img_plabels, labels)
    ami = adjusted_mutual_info_score(img_plabels, labels)

    return ari, ami


def zero_shot_eval(model, zeroshot_dataset, epoch, preprocess, args):

    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    
    # # TODO: remove nori dependency
    # if zeroshot_dataset=='imagenet':
    #    dataset = ImageNet_nori(transform=preprocess, split='val')
    # elif zeroshot_dataset=='cifar10':
    #     dataset = CIFAR10(root=args.datasets_dir, download=True, train=False, transform=preprocess)
    # elif zeroshot_dataset=='cifar100':
    #     dataset = CIFAR100(root=args.datasets_dir, download=True, train=False, transform=preprocess)
    # elif zeroshot_dataset=='stl10':
    #     dataset = STL10(root=args.datasets_dir, download=True, split='test', transform=preprocess)
    # else:
    #     # for ['birdsnap', 'country211', 'flowers102', 'gtsrb', 'stanford_cars', 'ucf101']
    #     data_path = f'{args.datasets_dir}/{zeroshot_dataset}/test'
    #     if zeroshot_dataset == 'ucf101':
    #         data_path += 'list01'
    #     logging.info(f'Loading data from  {data_path}')

    #     dataset = torchvision.datasets.ImageFolder(data_path, transform=preprocess)
    
    dataset = get_dataset(dataset_name=zeroshot_dataset, split='test', root=args.datasets_dir, transform=preprocess)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.evaluation_workers)

    mean_per_class_datasets = [
        'FGVCAircraft',
        'OxfordIIITPet',
        'Caltech101',
        'Flowers102',
        'flowers102'
    ]


    logging.info(f'Calculating text classifier for {zeroshot_dataset}')
    classnames, prompt_templates = dataset.classes, dataset.templates
    import copy
    classnames = copy.deepcopy(classnames)
    if zeroshot_dataset == 'birdsnap':
        # https://github.com/ml-jku/cloob/issues/10
        # FileNotFoundError: Found no valid file for the classes 046, 066, 123, 299, 302, 351, 403, 436, 465
        # these empty folders are removed
        empty_indexs = [46, 66, 123, 299, 302, 351, 403, 436, 465]
        for empty_index in empty_indexs[::-1]:
            del classnames[empty_index]
    
    classifier = zero_shot_classifier(model, classnames, prompt_templates, args)

    logging.info(f'Calculating image features for {zeroshot_dataset}')
    results = {}
    acc, features, labels = run(model, classifier, dataloader, args, mean_per_class = zeroshot_dataset in mean_per_class_datasets)
    logging.info(f'{zeroshot_dataset} zero-shot accuracy: {acc}%')
    results[f'{zeroshot_dataset}-zeroshot-acc'] = acc
    #results[f'{zeroshot_dataset}-zeroshot-accuracy-top5'] = top5

    # clustering evaluation
    # ari, ami = clustering_evaluation(features, labels)
    # logging.info(f'{zeroshot_dataset} clustering evaluation: ARI: {ari:.4f}, AMI:{ami:.4f}')
    # results[f'{zeroshot_dataset}-adjusted-rand-index'] = ari
    # results[f'{zeroshot_dataset}-adjusted-mutual-info'] = ami

    
    for key, item in results.items():
        results[key] = float(item)
    
    return results
