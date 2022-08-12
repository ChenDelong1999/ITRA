
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 32 --save-frequency 8 --batch-size 512 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --unlock-image-model --image-head-n-layers 2 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'sentence-transformers/all-roberta-large-v1' \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/Stage1' --copy-codebase --name 'U[RN50-h2]-[InfoNCE]-L[all-roberta-large-v1]-bs4096-32ep'

   
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 32 --save-frequency 8 --batch-size 256 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --unlock-image-model --image-head-n-layers 2 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'sentence-transformers/all-roberta-large-v1' \
    --distiller 'InfoNCE' --BYOL \
    --report-to tensorboard --logs 'logs/Stage1' --copy-codebase --name 'U[RN50-h2]-[InfoNCE-BYOL]-L[all-roberta-large-v1]-bs2048-32ep' 

# Datasets Zoo
    --dataset-size 14000000 --episode-size 2000000 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    