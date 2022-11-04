conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 32 --save-frequency 8 --batch-size 512 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'mobilenet_v3_small' --image-model-builder 'torchvision' --unlock-image-model --image-head-n-layers 2 \
    --text-model 'RN50' --text-model-builder 'OpenCLIP' --pretrained-text-model --text-head-n-layers 0 \
    --distiller 'InfoNCE' --teaher 'text' \
    --report-to tensorboard --logs 'logs/SmallCLIP' --copy-codebase --name 'U[mobilenet_v3_small-h2]-[InfoNCE]-L[CLIP-RN50]-bs4096-YFCC-32ep'



    # ####  # ####  # ####  # ####  # ####  # ####  # ####  # ####  # ####

# KD
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 100000 --train-data 'cache/yfcc_nori.csv' \
    --epochs 50 --save-frequency 50 --batch-size 64 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 0  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' --eval-first \
    --lr 1e-4 --warmup 100 --wd 0 --eps 1e-6 --max-grad-norm 1 \
    --image-model 'ViT-B-16' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 0 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 3 --unlock-text-model  --text-model 'roberta-base' \
    --text-pooler 'cls' --max-seq-length 32 --logit-scale 0.07 \
    --distiller 'DINO' --dino-teacher-temp 0.02 --teacher 'image' --w-simcse 0 --w-distill 1  \
    --report-to tensorboard --logs 'logs/V2L-KD-1031' --copy-codebase --name 'roberta_base-KD(DINO-0.02)-bs512-YFCC-1Mx5-lr1e-4'

# KD
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 100000 --train-data 'cache/yfcc_nori.csv' \
    --epochs 50 --save-frequency 50 --batch-size 64 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 0  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' --eval-first \
    --lr 1e-4 --warmup 100 --wd 0 --eps 1e-6 --max-grad-norm 1 \
    --image-model 'ViT-B-16' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 0 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 3 --unlock-text-model  --text-model 'roberta-base' \
    --text-pooler 'cls' --max-seq-length 32 --logit-scale 0.07 \
    --distiller 'DINO' --dino-teacher-temp 0.025 --teacher 'image' --w-simcse 0 --w-distill 1  \
    --report-to tensorboard --logs 'logs/V2L-KD-1031' --copy-codebase --name 'roberta_base-KD(DINO-0.025)-bs512-YFCC-1Mx5-lr1e-4'

# KD
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 100000 --train-data 'cache/yfcc_nori.csv' \
    --epochs 50 --save-frequency 50 --batch-size 64 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 0  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' --eval-first \
    --lr 1e-4 --warmup 100 --wd 0 --eps 1e-6 --max-grad-norm 1 \
    --image-model 'ViT-B-16' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 0 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 3 --unlock-text-model  --text-model 'roberta-base' \
    --text-pooler 'cls' --max-seq-length 32 --logit-scale 0.07 \
    --distiller 'DINO' --dino-teacher-temp 0.03 --teacher 'image' --w-simcse 0 --w-distill 1  \
    --report-to tensorboard --logs 'logs/V2L-KD-1031' --copy-codebase --name 'roberta_base-KD(DINO-0.03)-bs512-YFCC-1Mx5-lr1e-4'

# KD
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 100000 --train-data 'cache/yfcc_nori.csv' \
    --epochs 50 --save-frequency 50 --batch-size 64 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 0  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' --eval-first \
    --lr 1e-4 --warmup 100 --wd 0 --eps 1e-6 --max-grad-norm 1 \
    --image-model 'ViT-B-16' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 0 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 3 --unlock-text-model  --text-model 'roberta-base' \
    --text-pooler 'cls' --max-seq-length 32 --logit-scale 0.07 \
    --distiller 'DINO' --dino-teacher-temp 0.035 --teacher 'image' --w-simcse 0 --w-distill 1  \
    --report-to tensorboard --logs 'logs/V2L-KD-1031' --copy-codebase --name 'roberta_base-KD(DINO-0.035)-bs512-YFCC-1Mx5-lr1e-4'
