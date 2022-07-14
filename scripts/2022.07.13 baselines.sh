# 4x2080ti  
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 64 --save-frequency 4 --batch-size 100 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 3 \
    --lr 5e-5 --warmup 2000 --wd 0.1 --max-grad-norm 10 \
    --distiller 'SEED' \
    --report-to tensorboard --logs logs/exp7_baselines --copy-codebase --name 'T[mpnet]_S[RN50]_[SEED]_bs400_lr5e-5_64ep(CC2.5M)'


# [Distiller Zoo]
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


# 8x2080ti  
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 50 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 3 \
    --lr 5e-5 --warmup 2000 --wd 0.1 --max-grad-norm 10 \
    --distiller 'RKD' --w-rkd-d 0 --w-rkd-a 1 \
    --report-to tensorboard --logs logs/exp7_baselines --copy-codebase --name 'T[mpnet]_S[RN50]_[RKD-A]_bs400_lr5e-5_16ep(CC2.5M)'

torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 50 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 3 \
    --lr 5e-5 --warmup 2000 --wd 0.1 --max-grad-norm 10 \
    --distiller 'RKD' --w-rkd-d 1 --w-rkd-a 1 \
    --report-to tensorboard --logs logs/exp7_baselines --copy-codebase --name 'T[mpnet]_S[RN50]_[RKD-DA]_bs400_lr5e-5_16ep(CC2.5M)'