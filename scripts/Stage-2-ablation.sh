
# Stage 2, Full fine-tuning (LiT) with V100
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 2000000 --train-data 'cache/yfcc_nori.csv' \
    --epochs 56 --save-frequency 56 --batch-size 32 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --image-head-n-layers 0 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --unlock-text-model --text-head-n-layers 0 --text-model 'sentence-transformers/all-roberta-large-v1' \
    --distiller 'InfoNCE' --restart --find-unused-parameters \
    --resume 'logs/Stage1/U[RN50-pretrained-h2]-[InfoNCE]-L[all-roberta-large-v1]-bs4096-8ep/checkpoints/epoch_8.pt' \
    --report-to tensorboard --logs 'logs/Stage2-adapter-ablation-(resume-8ep)' --copy-codebase --name 'L[RN50-h0]_[InfoNCE]_U[all-roberta-large-v1]-bs256-8ep'

# Stage 2, adaptive-tuning
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 2000000 --train-data 'cache/yfcc_nori.csv' \
    --epochs 56 --save-frequency 56 --batch-size 32 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --image-head-n-layers 0 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'sentence-transformers/all-roberta-large-v1' \
    --distiller 'InfoNCE' --restart --adapter 'prefix_tuning' --n-prompt 1 \
    --resume 'logs/Stage1/U[RN50-pretrained-h2]-[InfoNCE]-L[all-roberta-large-v1]-bs4096-8ep/checkpoints/epoch_8.pt' \
    --report-to tensorboard --logs 'logs/Stage2-adapter-ablation-(resume-8ep)' --copy-codebase --name 'L[RN50-h0]_[InfoNCE]_L[all-roberta-large-v1]-bs256-8ep-prefix_tuning_1'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# adapters
    --adapter 'prefix_tuning' --n-prompt 16
    --adapter 'bottleneck_adapter' 
    --adapter 'lang_adapter'
    --adapter 'dummy'
    --adapter 'mam_adapter'
