
# Stage 1, from x-transformer
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 0 --train-data 'cache/yfcc_nori.csv' \
    --epochs 32 --save-frequency 32 --batch-size 64 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --unlock-image-model --image-head-n-layers 2 \
    --text-model-builder 'sentence-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'all-mpnet-base-v2' \
    --distiller 'InfoNCE' --BYOL \
    --report-to tensorboard --logs 'logs/8xV100-YFCC14M-32ep' --copy-codebase --name 'U[RN50-h2]-L[all-mpnet-base-v2]-bs4096-32ep-BYOL'



    

# 8x2080ti YFCC-14M 8 epoch
# Stage 2, LiT-tuning
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 1 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 56 --save-frequency 56 --batch-size 64 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 20000 --wd 0.5 --max-grad-norm 1 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --pretrained-image-model  --image-head-n-layers 2 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'sentence-transformers/all-roberta-large-v1' \
    --distiller 'InfoNCE' --restart --find-unused-parameters --adapter 'prefix_tuning' \
    --report-to tensorboard --logs 'logs/8x2080ti-YFCC14M-8ep-LiT' --copy-codebase --name 'L[RN50-h2]_L[all-mpnet-base-v2-h0]-prompt-bs512'



# Stage 2, prefix-tuning， random
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 2000000 --train-data 'cache/yfcc_nori.csv' \
    --epochs 56 --save-frequency 56 --batch-size 64 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 20000 --wd 0.5 --max-grad-norm 1 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --image-head-n-layers 0 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'sentence-transformers/all-roberta-large-v1' \
    --distiller 'InfoNCE' --restart --adapter 'prefix_tuning' --n-prompt 4 --find-unused-parameters \
    --report-to tensorboard --logs 'logs/Stage2' --copy-codebase --name 'L[RN50-random-h0]_L[all-roberta-large-v1]-bs512-8ep-prefix(4)tuning'