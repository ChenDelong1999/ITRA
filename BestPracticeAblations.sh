# Adapters

conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="$PYTHONPATH:$PWD/src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --epochs 14 --save-frequency 14 --batch-size 100 \
    --dataset-size 14000000 --episode-size 1000000 --train-data 'cache/yfcc_nori.csv' \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model-builder 'openclip' --image-model 'RN50' --pretrained-image-model --image-head-n-layers 0 \
    --text-model 'bert-base-uncased' --text-model-builder 'huggingface' --adapter 'lang_adapter' --text-head-n-layers 2 \
    --distiller 'InfoNCE' --lock-image-model \
    --report-to tensorboard --logs 'logs/BestPractice-Adapter' --cache-dir 'cache/weights' --name 'bottleneck_adapter_lr1e-4'

--adapter 'bottleneck_adapter'
--adapter 'lang_adapter'
--adapter 'prefix_tuning'
--adapter 'dummy'
--adapter 'lora_adapter'
--adapter 'ia3_adapter'
--adapter 'mam_adapter'
--adapter 'unipelt'