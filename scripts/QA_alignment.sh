
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 4000000 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 28 --save-frequency 28 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'ViT-B-16' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 3 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'sentence-transformers/multi-qa-MiniLM-L6-dot-v1' --text-pooler 'mean' \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/QA_alignment' --copy-codebase --name 'L[ViT-B-16-h3]-L[multi-qa-MiniLM-L6-dot-v1]-bs1024-8ep'