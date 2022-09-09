import os
import tqdm
import pyarrow.parquet as pq
import refile
import tarfile
import pickle
import sys
# oss cp --recursive 's3://zyz/laion400m_full/nori_lists/' '<nori_list_dir>'

nori_list_dir = '/data/Datasets/laion400m/'
full_dataset = 's3://collect-22040715001-data/'

x_part = sys.argv[1]

caption_dict = {}
dict_file = f'cache/laion400m_caption_dict_part_{x_part}.pkl'
#for x_part in range(4):
for x_tar in range(10000):
    x_tar_str = str(x_part) + str(x_tar).zfill(4)
    if x_part==3 and x_tar>1455:
        continue
    tar = tarfile.open(fileobj=refile.smart_open(full_dataset+f'laion400m_part{x_part}/{x_tar_str}.tar', "rb"))
    print(f'Processing {x_tar} of 10000 ({round(x_tar/10000,2)}%)', full_dataset+f'laion400m_part{x_part}/{x_tar_str}.tar')
    files = tar.getnames()
    for file in tqdm.tqdm(files):
        if '.txt' in file:
            x_jpg = file.replace('.txt', '')
            caption = tar.extractfile(file).read()
            caption_dict[f'{x_part}-{x_tar_str}-{x_jpg}'] = caption

    with open(dict_file, "wb") as tf:
        pickle.dump(caption_dict, tf)


exit()

# def get_caption(x_part, x_tar, x_jpg):
#     tar = tarfile.open(fileobj=refile.smart_open(full_dataset+f'laion400m_part{x_part}/{x_tar}.tar', "rb"))
#     text = tar.extractfile(f"{x_jpg}.txt").read()
#     return text

nori_lists = os.listdir(nori_list_dir)
samples = []
for i in tqdm.tqdm(range(len(nori_lists))):
    file = nori_list_dir + nori_lists[i]
    lines = open(file).readlines()
    for line in lines:
        line = line.strip().split('\t')
        nori_id, path = line
        x_part, x_tar, x_jpg = path.split('/')
        x_tar = x_tar.replace('.tar', '')
        x_jpg = x_jpg.replace('.jpg', '')
        caption = get_caption(x_part, x_tar, x_jpg)
        print(nori_id, path, caption)
        samples.append([nori_id, path, caption])
    print(len(samples))

 