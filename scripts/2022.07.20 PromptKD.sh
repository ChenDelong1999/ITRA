"""
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 32 --save-frequency 4 --batch-size 64 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 100 \
    --model RN50 --projection-n-layers 0 --augmentation protoclip-light-augmentation \
    --text-teacher 'RN50' --unlock-text-teacher --random-text-teacher \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs logs/exp10_PromptedKD --copy-codebase --name 'T[CLIP-random-unlock]_S[RN50]_[InfoNCE]_bs512_lr1e-4_32ep(CC2.5M)'
"""

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 32 --save-frequency 4 --batch-size 64 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --text-lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 100 \
    --model 'RN50'  --projection-n-layers 0 --augmentation protoclip-light-augmentation \
    --text-teacher 'RN50'  --prompt --n-prompt 8 \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/exp10_PromptedKD_8x2080' --copy-codebase --name 'T[RN50-pretrained-lock-prompted(8)]_S[RN50]_[InfoNCE]_bs512_lr1e-4_32ep(CC2.5M)'

    --text-teacher 'RN50' --unlock-text-teacher --random-text-teacher \RN50-random
    --text-teacher 'RN50' --unlock-text-teacher \RN50-pretrained-unlock
    --text-teacher 'RN50' \RN50-pretrained-lock
    --text-teacher 'RN50'  --prompt --n-prompt 1 \RN50-pretrained-lock-prompted(1)
    --text-teacher 'RN50'  --prompt --n-prompt 4 \RN50-pretrained-lock-prompted(4)
    --text-teacher 'RN50'  --prompt --n-prompt 8 \RN50-pretrained-lock-prompted(8)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# [Datasets]
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --dataset-size 14000000 --episode-size 2500000 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \

# [Text Teachers]
    --text-teacher 'all-mpnet-base-v2' --projection-dim 768 \
    --text-teacher 'average_word_embeddings_glove.6B.300d' --projection-dim 300 \
    --text-teacher 'clip-ViT-B-32'  --projection-dim 512 \
    --text-teacher 'clip-ViT-B-16'  --projection-dim ? \
    --text-teacher 'clip-ViT-L-14'  --projection-dim ? \

# [Image Teachers]
    --image-teacher 'RN50'
    --image-teacher 'resnet50'

# [Students]
## OpenCLIP model
    --model RN50 --open-clip-model \
    --model RN50 --open-clip-model --pretrained openai \
    --model RN50 --open-clip-model --pretrained openai --freeze-student-backbone \

    --model RN101 --open-clip-model
    --model RN50x4 --open-clip-model
    --model RN50x16 --open-clip-model
    --model ViT-B-32 --open-clip-model
    --model ViT-B-16 --open-clip-model
    --model ViT-H-14 --open-clip-model
    --model ViT-L-14 --open-clip-model

    --model resnet50
    --model resnet50 --pretrained 'torchvision'


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
