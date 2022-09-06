# Stage 2, adaptive-tuning
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 32 --save-frequency 8 --batch-size 128 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-5 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 2 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'sentence-transformers/all-roberta-large-v1' \
    --distiller 'InfoNCE' --restart --adapter 'bottleneck_adapter' \
    --resume 'logs/Stage1/U[RN50-h2]-[InfoNCE]-L[all-roberta-large-v1]-bs4096-32ep/checkpoints/epoch_32.pt' \
    --report-to tensorboard --logs 'logs/Stage2-from-U[RN50-h2]-[InfoNCE]-L[all-roberta-large-v1]-bs4096-32ep' --copy-codebase --name 'L[RN50-h2]_[InfoNCE]_L[all-roberta-large-v1]-bs1024-32ep-lr5e-5-bottleneck_adapter'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# adapters
    --adapter 'prefix_tuning' --n-prompt 16
    --adapter 'bottleneck_adapter' 
    --adapter 'lang_adapter'
    --adapter 'dummy'
    --adapter 'mam_adapter'
