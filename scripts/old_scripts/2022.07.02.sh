# average_word_embeddings_glove.6B.300d, 8卡训练 4，8，16，32 epoch

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 4 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 300 --projection-n-layers 2 --pretrained-text 'average_word_embeddings_glove.6B.300d' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[average_word_embeddings_glove.6B.300d]_stu[RN50]_[MSE]_bs1024_lr5e-4_4-epochs(CC250w)'

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 8 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 300 --projection-n-layers 2 --pretrained-text 'average_word_embeddings_glove.6B.300d' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[average_word_embeddings_glove.6B.300d]_stu[RN50]_[MSE]_bs1024_lr5e-4_8-epochs(CC250w)'

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 300 --projection-n-layers 2 --pretrained-text 'average_word_embeddings_glove.6B.300d' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[average_word_embeddings_glove.6B.300d]_stu[RN50]_[MSE]_bs1024_lr5e-4_16-epochs(CC250w)'

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 32 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 300 --projection-n-layers 2 --pretrained-text 'average_word_embeddings_glove.6B.300d' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[average_word_embeddings_glove.6B.300d]_stu[RN50]_[MSE]_bs1024_lr5e-4_32-epochs(CC250w)'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# all-MiniLM-L6-v2, 4卡训练 16，32 epoch
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 384 --projection-n-layers 2 --pretrained-text 'all-MiniLM-L6-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-MiniLM-L6-v2]_stu[RN50]_[MSE]_bs512_lr25e-6_16-epochs(CC250w)'

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 32 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 384 --projection-n-layers 2 --pretrained-text 'all-MiniLM-L6-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-MiniLM-L6-v2]_stu[RN50]_[MSE]_bs512_lr25e-6_32-epochs(CC250w)'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# all-mpnet-base-v2, 4卡训练 16，32 epoch
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 768 --projection-n-layers 2 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[MSE]_bs512_lr25e-6_16-epochs(CC250w)'

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 32 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 768 --projection-n-layers 2 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[MSE]_bs512_lr25e-6_32-epochs(CC250w)'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# all-mpnet-base-v2, 4卡训练 16，32 epoch，YFCC-14M，eposodic size=250w
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 14000000 --episode-size 2500000 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 16 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 768 --projection-n-layers 2 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[MSE]_bs512_lr25e-6_16-epochs(YFCC250w)'

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 14000000 --episode-size 2500000 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 32 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 768 --projection-n-layers 2 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[MSE]_bs512_lr25e-6_32-epochs(YFCC250w)'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# all-mpnet-base-v2, 4卡训练 16 epoch, projection-n-layers=4/1/3
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 768 --projection-n-layers 4 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[MSE]_bs512_lr25e-6_16-epochs(CC250w)_head4layer'

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 768 --projection-n-layers 1 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[MSE]_bs512_lr25e-6_16-epochs(CC250w)_head1layer'

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 768 --projection-n-layers 3 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[MSE]_bs512_lr25e-6_16-epochs(CC250w)_head3layer'

# 2022.07.04
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 768 --projection-n-layers 5 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[MSE]_bs512_lr25e-6_16-epochs(CC250w)_head5layer'

    
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 768 --projection-n-layers 6 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[MSE]_bs512_lr25e-6_16-epochs(CC250w)_head6layer'
    
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 768 --projection-n-layers 7 --pretrained-text 'all-mpnet-base-v2' \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 0.01 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[MSE]_bs512_lr25e-6_16-epochs(CC250w)_head7layer'