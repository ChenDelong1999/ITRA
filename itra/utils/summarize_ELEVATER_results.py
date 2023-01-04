import os
import numpy as np
import pandas as pd



# log_dir = '/data/codes/ProtoRKD/logs/codebase_test/U[mobilenet_v3_large-h2]-L[CLIP-from-RN50]-bs1024-YFCC-56ep-lr1e-5/ELEVATER_evaluation/5-shot/log'

mode_dir = input('Input your log dir (end with "../ELEVATER_evaluation/<eval_mode>"):\n>>> ')
mode = mode_dir.split('/')[-1]
log_dir = os.path.join(mode_dir, 'log')
datasets = os.listdir(log_dir)

all_datasets = []
all_accuracies = []

for dataset in datasets:
    if dataset in ['predictions', 'summary.csv']:
        continue
    log_file = os.listdir(os.path.join(log_dir, dataset))[-1]

    line = open(os.path.join(log_dir, dataset, log_file)).readlines()[-1]

    all_datasets.append(dataset)
    try:
        all_accuracies.append(float(line.split()[-1].replace('%', '')))
    except:        
        all_accuracies.append(0)

all_datasets.append('Average')
all_accuracies.append(np.mean(all_accuracies))

df = pd.DataFrame(
    data={
        'Dsataset': all_datasets,
        f'{mode}-accuracy%':all_accuracies
    }
)
print(df)
csv_file = os.path.join(mode_dir, 'summary.csv')
df.to_csv(csv_file)
print(f'saved to {csv_file}')
