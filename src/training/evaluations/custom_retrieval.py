import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from clip_benchmark.metrics.zeroshot_retrieval import recall_at_k, batchify


from training.data import CsvDataset

def custom_retrieval_evaluation(model, epoch, preprocess, args, recall_k_list=[1,5,10]):
 
    """
    Modified from https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/metrics/zeroshot_retrieval.py
    Evaluate the model on the given dataset
    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda
    recall_k_list: list of int
        recall@k k's to use
    
    Returns
    -------
    
    dict of retrieval metrics
    """

    if args.retrieval_frequency == 0:
        return {}
    if (epoch % args.retrieval_frequency) != 0 and epoch != args.epochs:
        return {}
        
    dataset = CsvDataset(
        input_filename=args.retrieval_data,
        transforms=preprocess,
        img_key=args.retrieval_csv_img_key,
        caption_key=args.retrieval_csv_caption_key,
        sep=args.retrieval_csv_separator,
        nori_dataset=args.retrieval_nori_dataset,
        images_dir=os.path.join(args.eval_data_dir, args.retrieval_images_dir)
    )

    positive_pairs = np.zeros([len(dataset), len(dataset)])
    for i in range(len(dataset)):
        positive_texts = np.where(dataset.images==dataset.images[i])
        for positive_text in positive_texts:
            positive_pairs[i, positive_text] = 1
        positive_images = np.where(dataset.captions==dataset.captions[i])
        for positive_image in positive_images:
            positive_pairs[positive_image, i] = 1

    positive_pairs = torch.from_numpy(positive_pairs)


    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    # texts_image_index = []
    for _, batch_images, batch_texts in tqdm(dataloader):
        batch_images = batch_images.to(args.device)
        # batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        # compute the embedding of images and texts
        with torch.no_grad():
            if args.distributed and not args.horovod:
                batch_text_features = model.module.encode_text(batch_texts, projection=True)
                batch_image_features = model.module.encode_image(batch_images, projection=True)
            else:
                batch_text_features = model.encode_text(batch_texts, projection=True)
                batch_image_features = model.encode_image(batch_images, projection=True)

            batch_images_emb = F.normalize(batch_text_features, dim=-1)
            batch_texts_emb = F.normalize(batch_image_features, dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        
    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    images_emb = images_emb / images_emb.norm(dim=-1, keepdim=True)
    texts_emb = texts_emb / texts_emb.norm(dim=-1, keepdim=True)
        

    # get the score for each text and image pair
    scores  = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    # positive_pairs = torch.zeros_like(scores, dtype=bool)
    # positive_pairs[torch.arange(len(scores)), torch.arange(len(scores))] = True
    metrics = {}
    for recall_k in recall_k_list:
        '''
        Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        for each image, that number will be greater than 1 for text retrieval.
        However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        it over the dataset.
        '''
        metrics[f"custom-text2image-R@{recall_k}"] = round((batchify(recall_at_k, scores, positive_pairs, batch_size, args.device, k=recall_k)>0).float().mean().item() * 100, 2)
        metrics[f"custom-image2text-R@{recall_k}"] = round((batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, args.device, k=recall_k)>0).float().mean().item() * 100, 2)

    metrics[f"custom-mean-recall"] = round(np.mean(list(metrics.values())), 2)


    return metrics
    

