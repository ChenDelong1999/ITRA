
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model resnet50 --pretrained 'torchvision' \
    --add-projection-head --projection-dim 768 --projection-n-layers 4 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-5 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'RKD' --w-rkd-d 1 --w-rkd-a 0 \
    --report-to tensorboard --logs logs/exp5_freeze_pretrained_student --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[resnet50-torchvision-pretrained]_[RKD-D]_bs640_lr25e-5_16-epochs(CC250w)'


conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 160 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model resnet50 --pretrained 'torchvision' \
    --add-projection-head --projection-dim 768 --projection-n-layers 4 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-5 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'SimReg' \
    --report-to tensorboard --logs logs/exp5_freeze_pretrained_student --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[resnet50-torchvision-pretrained]_[SimReg]_bs1280_lr25e-5_16-epochs(CC250w)'
