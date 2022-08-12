
# SimReg baseline: CC2.5M 16 epochs [all-mpnet-base-v2]
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 0 --retrieval-frequency 0  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --add-projection-head --projection-dim 768 --projection-n-layers 4 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --distiller 'SimReg' \
    --report-to tensorboard --logs logs/debug --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[SimReg]_bs320_lr25e-6_16-epochs(CC250w)'

# SimReg baseline: CC2.5M 16 epochs [average_word_embeddings_glove.6B.300d]
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 0 --retrieval-frequency 0  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --add-projection-head --projection-dim 300 --projection-n-layers 4 --pretrained-text 'average_word_embeddings_glove.6B.300d' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --distiller 'SimReg' \
    --report-to tensorboard --logs logs/exp3_baselines --copy-codebase --name 'tea[average_word_embeddings_glove.6B.300d]_stu[RN50]_[SimReg]_bs320_lr25e-6_16-epochs(CC250w)'


# 32 epochs - - - 

# SimReg baseline: CC2.5M 32 epochs [all-mpnet-base-v2]
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 32 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 0 --retrieval-frequency 0  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --add-projection-head --projection-dim 768 --projection-n-layers 4 --pretrained-text 'all-mpnet-base-v2' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --distiller 'SimReg' \
    --report-to tensorboard --logs logs/exp3_baselines --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[SimReg]_bs640_lr25e-6_32-epochs(CC250w)'

# SimReg baseline: CC2.5M 32 epochs [average_word_embeddings_glove.6B.300d]
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 32 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 0 --retrieval-frequency 0  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --add-projection-head --projection-dim 300 --projection-n-layers 4 --pretrained-text 'average_word_embeddings_glove.6B.300d' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --distiller 'SimReg' \
    --report-to tensorboard --logs logs/exp3_baselines --copy-codebase --name 'tea[average_word_embeddings_glove.6B.300d]_stu[RN50]_[SimReg]_bs640_lr25e-6_32-epochs(CC250w)'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# RKD   
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 0 --retrieval-frequency 0  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --add-projection-head --projection-dim 768 --projection-n-layers 4 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --distiller 'RKD' --w-rkd-d 1 --w-rkd-a 0 \
    --report-to tensorboard --logs logs/exp3_baselines --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[RKD-D]_bs320_lr25e-6_16-epochs(CC250w)'
    
# RKD baseline: CC2.5M 16 epochs [average_word_embeddings_glove.6B.300d]
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 0 --retrieval-frequency 0  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --add-projection-head --projection-dim 300 --projection-n-layers 4 --pretrained-text 'average_word_embeddings_glove.6B.300d' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --distiller 'RKD' \
    --report-to tensorboard --logs logs/exp3_baselines --copy-codebase --name 'tea[average_word_embeddings_glove.6B.300d]_stu[RN50]_[RKD]_bs320_lr25e-6_16-epochs(CC250w)'

