
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 0 --retrieval-frequency 0  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --add-projection-head --projection-dim 768 --projection-n-layers 4 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CompRess' \
    --report-to tensorboard --logs logs/exp4_CompRess --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[CompRess]_bs320_lr25e-6_16-epochs(CC250w)'


conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 0 --retrieval-frequency 0  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --add-projection-head --projection-dim 768 --projection-n-layers 4 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CompRess-1q' \
    --report-to tensorboard --logs logs/exp4_CompRess --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[CompRess-1q]_bs320_lr25e-6_16-epochs(CC250w)'


conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 0 --retrieval-frequency 0  --eval-data-dir '/data/Datasets' \
    --model alexnet --pretrained torchvision \
    --add-projection-head --projection-dim 768 --projection-n-layers 2 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CompRess' \
    --report-to tensorboard --logs logs/exp4_CompRess --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[alexnet-torchvision-pretrained]_[CompRess]_bs320_lr25e-6_16-epochs(CC250w)'
