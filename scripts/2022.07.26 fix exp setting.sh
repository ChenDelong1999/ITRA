# 2022.07.22 exp plan (7x8x2080ti)
- mobilenet v3 small (16ep)
    # alexnet 惨不忍睹，换一个好点得小模型 
    1. [mobilenet random baseline]
    2. pretrained CLIP text - lock
    3. mpnet - lock
    4. mpnet - unlock (collapsed)
- MLP head
    # MLP projection head 到底有没有用
    1. [alexnet random baseline] - 3xMLP
    2. RN50 - mpnet - lock - 3 x image MLP
    3. RN50 - mpnet - lock - 3 x text MLP
- YFCC 32 ep
    # 到底是不是过拟合
    1. RN50 - mpnet - lock

# YFCC
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 2500000 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 64 --save-frequency 8 --batch-size 64 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --unlock-image-model --image-head-n-layers 7 \
    --text-model 'all-mpnet-base-v2' --text-model-builder 'sentence-transformer' --pretrained-text-model --text-head-n-layers 0 \
    --distiller 'InfoNCE'\
    --report-to tensorboard --logs 'logs/exp12_lock_mpnet_YFCC' --copy-codebase --name 'T[mpnet-lock]_S[RN50(7)]_[InfoNCE]_bs512_lr5e-4_64ep(YFCC2.5M)'

# resume YFCC
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 2500000 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 64 --save-frequency 8 --batch-size 64 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --image-head-n-layers 2 \
    --text-model 'all-mpnet-base-v2' --text-model-builder 'sentence-transformer' --unlock-text-model --pretrained-text-model --text-head-n-layers 0 \
    --distiller 'InfoNCE'\
    --resume 'logs/exp12_lock_mpnet_YFCC/T[mpnet-lock(0)]_S[RN50(2)]_[InfoNCE]_bs512_lr5e-4_32ep(YFCC2.5M)/checkpoints/epoch_32.pt' \
    --report-to tensorboard --logs 'logs/exp12_lock_mpnet_YFCC' --copy-codebase --name 'T[mpnet-lock(0)]_S[RN50(2)-LiT]_[InfoNCE]_bs512_lr5e-4_32:64ep(YFCC2.5M)'

# resume Conceptual Captions
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 32 --save-frequency 4 --batch-size 64 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP'  --image-head-n-layers 0 \
    --text-model 'RN50' --unlock-text-model --text-head-n-layers 0 \
    --distiller 'InfoNCE'\
    --resume 'logs/exp11_refract/T[RN50-random]_S[RN50]_[InfoNCE]_bs512_lr5e-4_16ep(CC2.5M)/checkpoints/epoch_16.pt' \
    --report-to tensorboard --logs 'logs/exp11_refract' --copy-codebase --name 'T[RN50-random]_S[RN50(LiT)]_[InfoNCE]_bs512_lr5e-4_16:32ep(CC2.5M)'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# text model
    --text-model 'RN50' --unlock-text-model --text-head-n-layers 0 \RN50-random
    --text-model 'RN50' --pretrained-text-model --unlock-text-model --text-head-n-layers 0 \RN50-pretrained-unlock
    --text-model 'RN50' --pretrained-text-model --text-head-n-layers 0 \RN50-pretrained-lock
    
    --text-model 'all-mpnet-base-v2' --text-model-builder 'sentence-transformer' --pretrained-text-model --text-head-n-layers 0  --unlock-text-model \mpnet-unlock
    --text-model 'all-mpnet-base-v2' --text-model-builder 'sentence-transformer' --pretrained-text-model --text-head-n-layers 0 \mpnet-lock

    # 'average_word_embeddings_glove.6B.300d'
    #  --joint-projection-dim 1024 

# image model
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --unlock-image-model --image-head-n-layers 0 \RN50
    --image-model 'alexnet' --image-model-builder 'torchvision' --unlock-image-model --image-head-n-layers 1 \alexnet
    --image-model 'mobilenet_v3_small' --image-model-builder 'torchvision' --unlock-image-model --image-head-n-layers 1 \mobilenet_v3_small
    # "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"


# [Datasets]
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --dataset-size 14000000 --episode-size 2500000 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \



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
