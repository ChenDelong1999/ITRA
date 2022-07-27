# 8x2080ti YFCC-14M 8 epoch


# from scratch CLIP baseline
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 2000000 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 56 --save-frequency 14 --batch-size 64 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 20000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --unlock-image-model --image-head-n-layers 0 \
    --text-model 'RN50' --unlock-text-model --text-head-n-layers 0 \
    --distiller 'InfoNCE'\
    --report-to tensorboard --logs 'logs/8x2080ti-YFCC14M-8ep' --copy-codebase --name 'CLIP-RN50-bs512'


# Stage 1, from x-transformer
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 2000000 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 35 --save-frequency 35 --batch-size 64 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 20000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --unlock-image-model --image-head-n-layers 2 \
    --text-model-builder 'sentence-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'all-mpnet-base-v2' \
    --distiller 'InfoNCE'\
    --report-to tensorboard --logs 'logs/8x2080ti-YFCC14M-8ep' --copy-codebase --name '[:5]all-mpnet-base-v2(lock)-RN50-bs512'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# text model (--unlock-text-model)
    # OpenCLIP
    --text-model 'RN50' --unlock-text-model --text-head-n-layers 0 \RN50-random
    --text-model 'RN50' --pretrained-text-model --unlock-text-model --text-head-n-layers 0 \RN50-pretrained-unlock
    --text-model 'RN50' --pretrained-text-model --text-head-n-layers 0 \RN50-pretrained-lock

    # huggingface-transformer
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'microsoft/mpnet-base' \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'bert-base-uncased' \
    
    # sentence-transformer
    --text-model-builder 'sentence-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'all-mpnet-base-v2' \
    --text-model-builder 'sentence-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'average_word_embeddings_glove.6B.300d' \

# image model
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --unlock-image-model --image-head-n-layers 0 \RN50
    --image-model 'alexnet' --image-model-builder 'torchvision' --unlock-image-model --image-head-n-layers 1 \alexnet
    --image-model 'mobilenet_v3_small' --image-model-builder 'torchvision' --unlock-image-model --image-head-n-layers 1 \mobilenet_v3_small
    # "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"


# [Datasets]
    --dataset-size 14000000 --episode-size 2000000 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 56 --save-frequency 14 --batch-size 64 --workers 8 \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \


# [Distillers]
## SimReg
    --distiller 'SimReg' \
    --distiller 'SimReg-L1' \
    --distiller 'SimReg-SmoothL1' \

## RKD-D (RKD-DA CUDA OOM)
    --distiller 'RKD' --w-rkd-d 1 --w-rkd-a 0 \
    (angle loss CUDA OOM) --distiller 'RKD' --w-rkd-d 0 --w-rkd-a 1 \
    (angle loss CUDA OOM) --distiller 'RKD' --w-rkd-d 1 --w-rkd-a 1 \

## InfoNCE (CLIP)
    --distiller 'InfoNCE' \

## CompRess
    --distiller 'CompRess-1q' \
    --distiller 'CompRess-2q' \

## SEED
    --distiller 'SEED' \

## DINO
    --distiller 'DINO' \

## ProtoCPC
    --distiller 'ProtoCPC' \
