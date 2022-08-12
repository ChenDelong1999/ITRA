
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 0 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 1 --save-frequency 1 --batch-size 64 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 0  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --pretrained-image-model --image-model-builder 'OpenCLIP' --image-head-n-layers 1 \
    --text-model 'gpt2' --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 \
    --distiller 'InfoNCE'\
    --report-to tensorboard --logs 'logs/save_pretrained' --copy-codebase --name 'gpt2'