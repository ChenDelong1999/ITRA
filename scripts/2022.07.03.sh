# 100epoch
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 300 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 768 --projection-n-layers 4 --pretrained-text 'all-mpnet-base-v2' \
    --lr 5e-5 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --resume 'logs/tea[all-mpnet-base-v2]_stu[RN50]_[MSE]_bs1024_lr1e-4_100-epochs(CC250w)/checkpoints/epoch_100.pt' \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[MSE]_bs1024_lr1e-4_100:300-epochs(CC250w)'

# RN50x16
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 8 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50x16 --add-projection-head --projection-dim 768 --projection-n-layers 4 --pretrained-text 'all-mpnet-base-v2' \
    --lr 1e-5 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50x16]_[MSE]_bs64_lr1e-5_16-epochs(CC250w)'


# smaller data    
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 200000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 200 --save-frequency 50 --batch-size 128 --workers 8 \
    --linear-frequency 5  --zeroshot-frequency 5 --retrieval-frequency 5  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 768 --projection-n-layers 4 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[MSE]_bs512_lr25e-6_200-epochs(CC20w)'