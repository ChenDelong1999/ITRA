
# CLIP projection layers
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 3 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50-3proj]_[CLIP]_bs320_lr25e-6_16-epochs(CC250w)'

# CLIP projection layers
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 2 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50-2proj]_[CLIP]_bs320_lr25e-6_16-epochs(CC250w)'

# CLIP projection layers
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 1 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50-1proj]_[CLIP]_bs320_lr25e-6_16-epochs(CC250w)'

# CLIP projection layers
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 0 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50-0proj]_[CLIP]_bs320_lr25e-6_16-epochs(CC250w)'

# CLIP CLIP teacher
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'clip-ViT-B-32' \
    --add-projection-head --projection-dim 512 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[clip-ViT-B-32]_stu[RN50]_[CLIP]_bs320_lr25e-6_16-epochs(CC250w)'

# CLIP CLIP teacher
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'clip-ViT-B-16' \
    --add-projection-head --projection-dim 512 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[clip-ViT-B-16]_stu[RN50]_[CLIP]_bs320_lr25e-6_16-epochs(CC250w)'
