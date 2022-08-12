from training.evaluations.coco_retrieval import CocoCaptions, CocoDataset, CocoTexts
import tqdm
import os
import random
import pandas as pd

eval_data_dir = '/data/Datasets'
coco_val_root = os.path.join(eval_data_dir, 'coco2017/val2017')
coco_val_json = os.path.join(eval_data_dir, 'coco2017/annotations/captions_val2017.json')
coco_dataset = CocoCaptions(root=coco_val_root, annFile=coco_val_json)

all_samples = []
for i in tqdm.tqdm(range(len(coco_dataset))):
    all_samples.append(coco_dataset[i][1])

positive_pairs = []
negative_pairs = []
for s in range(len(all_samples)):
    n_text = len(all_samples[s])
    negative_candidates = list(range(len(all_samples)))
    negative_candidates.remove(s)
    for i in range(n_text):
        for j in range(i+1, n_text):
            positive_pairs.append([all_samples[s][i], all_samples[s][j]])
            neg_sample = random.choice(negative_candidates)
            random_text = random.randint(0, 4)
            negative_pairs.append([
                all_samples[s][random_text], 
                all_samples[neg_sample][random_text]]
                )

# print(positive_pairs)
# print(len(positive_pairs))

# print(negative_pairs)
# print(len(negative_pairs))

sentence1 = []
sentence2 = []
scores = []

for positive_pair in positive_pairs:
    sentence1.append(positive_pair[0].replace('\n', ''))
    sentence2.append(positive_pair[1].replace('\n', ''))
    scores.append(1)

for negative_pair in negative_pairs:
    sentence1.append(negative_pair[0].replace('\n', ''))
    sentence2.append(negative_pair[1].replace('\n', ''))
    scores.append(0)

df = pd.DataFrame({"score":scores, "sentence1":sentence1, "sentence2":sentence2})
df.to_csv('STS_coco_val2017.csv', sep='\t', index=None)
print(df)

