
# 先准备一个 stage 1 的结果
# 为了省时间，finetune 一个 head, 把 CLIP RN50 和 sentence-RoBERTa 对齐
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 8 --save-frequency 8 --batch-size 512 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --pretrained-image-model  --unlock-image-model --image-head-n-layers 2 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'sentence-transformers/all-roberta-large-v1' \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/Stage1' --copy-codebase --name 'U[RN50-pretrained-h2]-[InfoNCE]-L[all-roberta-large-v1]-bs4096-8ep'

 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Stage 2, LiT-tuning
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 2000000 --train-data 'cache/yfcc_nori.csv' \
    --epochs 56 --save-frequency 56 --batch-size 64 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 2e-5 --warmup 20000 --wd 0.5 --max-grad-norm 1 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 0 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --unlock-text-model --text-head-n-layers 0 --text-model 'sentence-transformers/all-roberta-large-v1' \
    --distiller 'InfoNCE' --restart --find-unused-parameters \
    --report-to tensorboard --logs 'logs/Stage2' --copy-codebase --name 'L[RN50-openai-h0]_U[all-roberta-large-v1]-bs256-8ep'

# Stage 2, prefix-tuning
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
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 0 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'sentence-transformers/all-roberta-large-v1' \
    --distiller 'InfoNCE' --restart --adapter 'lang_adapter' \
    --report-to tensorboard --logs 'logs/Stage2' --copy-codebase --name 'L[RN50-openai-h0]_L[all-roberta-large-v1]-bs512-8ep-lang_adapter'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# adapters
    --adapter 'prefix_tuning' --n-prompt 1
    --adapter 'bottleneck_adapter' 
    --adapter 'lang_adapter'
    --adapter 'dummy'
    --adapter 'mam_adapter'
