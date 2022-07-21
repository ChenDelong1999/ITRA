import torch
import logging
import numpy as np
import tqdm
import os
from torchvision.datasets.coco import CocoCaptions
from torch.utils.data import Dataset, DataLoader
from open_clip import tokenize as clip_tokenizer
from PIL import Image
from training.prompt import encode_text_with_prompt


class CocoDataset(Dataset):
    # modeified from https://github.com/uta-smile/TCL/blob/main/dataset/caption_dataset.py#L50
    # get the ground truth (1 image v.s. multiple captions, hiting each of them is ok) for retrieval
    def __init__(self, coco_dataset, coco_val_root, transform):        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.transform = transform
        
        txt_id = 0
        for index in range(len(coco_dataset)):
            ann_ids = coco_dataset.coco.getAnnIds(imgIds=coco_dataset.ids[index])
            anns = coco_dataset.coco.loadAnns(ann_ids)
            target = [ann['caption'] for ann in anns]

            path = coco_dataset.coco.loadImgs(coco_dataset.ids[index])[0]['file_name']
            path = os.path.join(coco_val_root, path)

            self.image.append(path)
            self.img2txt[index] = []

            for i, caption in enumerate(target):
                self.text.append(caption)
                self.img2txt[index].append(txt_id)
                self.txt2img[txt_id] = index
                txt_id += 1
        #self.text = torch.cat(self.text, dim=0)
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):   
        image_path = self.image[index]     
        image = Image.open(image_path).convert('RGB')     
        image = self.transform(image)  
        return image, index


class CocoTexts():
    def __init__(self, coco_dataset):  
        self.coco_dataset = coco_dataset
        
    def __len__(self):
        return len(self.coco_dataset.text)
    
    def __getitem__(self, index):    
        return self.coco_dataset.text[index]


def coco_retrieval_evaluation(model, epoch, preprocess, args):
    if args.retrieval_frequency == 0:
        return {}, None, None
    if (epoch % args.retrieval_frequency) != 0 and epoch != args.epochs:
        return {}, None, None

    coco_val_root = os.path.join(args.eval_data_dir, 'coco2017/val2017')
    coco_val_json = os.path.join(args.eval_data_dir, 'coco2017/annotations/captions_val2017.json')

    coco_dataset = CocoCaptions(root=coco_val_root, annFile=coco_val_json, transform=preprocess)
    coco_dataset = CocoDataset(coco_dataset, coco_val_root=coco_val_root, transform=preprocess)
    coco_retrieval_dataloader = DataLoader(
            coco_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.evaluation_workers, pin_memory=True, drop_last=False,
        )
    coco_dataset_text = CocoTexts(coco_dataset)
    coco_retrieval_text_dataloader = DataLoader(
            coco_dataset_text,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.evaluation_workers, pin_memory=True, drop_last=False,
        )

    with torch.no_grad():
        logging.info('extracting COCO text features...')
        all_text_features = []
            
        for texts in tqdm.tqdm(coco_retrieval_text_dataloader):    
            if args.distributed and not args.horovod:
                text_features = model.module.encode_text(texts, projection=True)
            else:
                text_features = model.encode_text(texts, projection=True)
            all_text_features.append(text_features.detach().cpu())
        all_text_features = torch.cat(all_text_features,dim=0)
        
        logging.info('extracting COCO image features...')
        all_image_features = []
        for images, img_id in tqdm.tqdm(coco_retrieval_dataloader): 
            images = images.to(args.device)

            if args.distributed and not args.horovod:
                image_features = model.module.encode_image(images, projection=True)
            else:
                image_features = model.encode_image(images, projection=True)
            all_image_features.append(image_features.detach().cpu())
        all_image_features = torch.cat(all_image_features,dim=0)

        # normalization, this step is important
        all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
        all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)
        
        scores_img2text = (all_image_features @ all_text_features.t()).detach()
        scores_text2img = scores_img2text.t().detach()
    
    retrieval_metrics = get_retrieval_metrics(
        scores_img2text.cpu().numpy(), 
        scores_text2img.cpu().numpy(),
        coco_retrieval_dataloader.dataset.img2txt,
        coco_retrieval_dataloader.dataset.txt2img
        )
    logging.info('COCO retrieval evaluation: '+ str(retrieval_metrics))

    deduplicated_text_features = torch.zeros_like(all_image_features)
    for i in range(len(coco_retrieval_dataloader.dataset.img2txt)):
        deduplicated_text_features[i] = all_text_features[coco_retrieval_dataloader.dataset.img2txt[i][0]]

    return retrieval_metrics, all_image_features, deduplicated_text_features


       
def get_retrieval_metrics(scores_img2text, scores_text2img, gt_img2text, gt_text2img):
    
    #Images->Text 
    ranks = np.zeros(scores_img2text.shape[0])
    for index,score in enumerate(scores_img2text):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        for i in gt_img2text[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    img2text_recall_at_1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    img2text_recall_at_5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    img2text_recall_at_10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_text2img.shape[0])
    for index,score in enumerate(scores_text2img):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == gt_text2img[index])[0][0]

    text2img_recall_at_1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    text2img_recall_at_5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    text2img_recall_at_10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (img2text_recall_at_1 + img2text_recall_at_5 + img2text_recall_at_10) / 3
    ir_mean = (text2img_recall_at_1 + text2img_recall_at_5 + text2img_recall_at_10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {
        'image2text-R@1': img2text_recall_at_1,
        'image2text-R@5': img2text_recall_at_5,
        'image2text-R@10': img2text_recall_at_10,
        #'image2text-R-mean': tr_mean,
        'text2image-R@1': text2img_recall_at_1,
        'text2image-R@5': text2img_recall_at_5,
        'text2image-R@10': text2img_recall_at_10,
        #'text2image-R-mean': ir_mean,
        'mean-recall': r_mean
        }
    
    for key, item in eval_result.items():
        eval_result[key] = float(item)
    return eval_result
