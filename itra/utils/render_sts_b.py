
import sys
from torch import autocast
import os
import pandas as pd
import gzip
import os
import csv
from sentence_transformers import  util, InputExample
import pandas as pd
import sys
from diffusers import StableDiffusionPipeline
import os
import numpy as np

rank = int(sys.argv[1])
world_size = 8
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
print(f'rank {rank} of {world_size}')


sts_dataset_path = os.path.join('/data/Datasets', 'stsbenchmark.tsv.gz')
if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

train_samples = []
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        sentence1 = row['sentence1']
        sentence2 = row['sentence2']
        #inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
        if row['split'] == 'dev':
            dev_samples.append([score, sentence1, sentence2])
        elif row['split'] == 'test':
            test_samples.append([score, sentence1, sentence2])
        else:
            train_samples.append([score, sentence1, sentence2])


model_id = "CompVis/stable-diffusion-v1-4"
device = f"cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

for i in range(len(test_samples)):
    if i % world_size==rank:
        score, sentence1, sentence2 = test_samples[i]
        print(f'{i}/{len(test_samples)}({int(i/len(test_samples)*100)}%)\t{score}\t{sentence1} | {sentence2}')
        for repeat in range(5):
            with autocast("cuda"):
                image_sentence1 = pipe(sentence1, guidance_scale=7.5)["sample"][0]  
                while np.sum(np.array(image_sentence1))==0:
                    print(f'regenerating {sentence1}')
                    image_sentence1 = pipe(sentence1, guidance_scale=7.5)["sample"][0]  

                image_sentence2 = pipe(sentence2, guidance_scale=7.5)["sample"][0]  
                while np.sum(np.array(image_sentence2))==0:
                    print(f'regenerating {sentence2}')
                    image_sentence2 = pipe(sentence2, guidance_scale=7.5)["sample"][0]  

            image_sentence1.save(f'/data/Datasets/rendered_sts/test/{sentence1.replace("/","-")}-{repeat}.png')
            image_sentence2.save(f'/data/Datasets/rendered_sts/test/{sentence2.replace("/","-")}-{repeat}.png')


