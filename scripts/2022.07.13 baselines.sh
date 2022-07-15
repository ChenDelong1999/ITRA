""" 4x2080ti standard templet, DON'T MODIFY!!!
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 100 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-5 --warmup 2000 --wd 0.1 --max-grad-norm 10 \
    --projection-n-layers 3 \
    --model RN50 --open-clip-model \
    --text-teacher 'all-mpnet-base-v2' --projection-dim 768 \
    --distiller 'SEED' \
    --report-to tensorboard --logs logs/exp7_baselines --copy-codebase --name 'T[mpnet]_S[RN50]_[SEED]_bs400_lr5e-5_16ep(CC2.5M)'
"""


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# [Datasets]
    --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \

# [Teachers]
    --text-teacher 'all-mpnet-base-v2' --projection-dim 768 \
    --text-teacher 'average_word_embeddings_glove.6B.300d' --projection-dim 300 \
    --text-teacher 'clip-ViT-B-32'  --projection-dim 512 \
    --text-teacher 'clip-ViT-B-16'  --projection-dim ? \
    --text-teacher 'clip-ViT-L-14'  --projection-dim ? \

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
