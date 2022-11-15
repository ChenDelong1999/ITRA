

# Sample Commands


Configureate environment

```bash
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="$PYTHONPATH:$PWD/src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
```

- **Single GPU, train a CLIP RN50 from scratch**

```bash
python src/training/main.py \
    --dataset-size 2500000 --episode-size 2500000 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 1 --save-frequency 1 --batch-size 100 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --unlock-image-model --image-model 'RN50' --text-model-builder 'openclip' \
    --unlock-text-model --text-model 'RN50' --image-model-builder 'openclip' \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/codebase_sample' --name 'Single-GPU-CLIP-from-scratch-CC2.5Mx1-epoch'
```
 
# **Best Practice: adapters**

```bash
torchrun --nproc_per_node 8 -m training.main \
    --epochs 14 --save-frequency 14 --batch-size 100 \
    --dataset-size 14000000 --episode-size 1000000 --train-data 'cache/yfcc_nori.csv' \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 100 --wd 0.5 --max-grad-norm 5 \
    --image-model-builder 'openclip' --image-model 'RN50' --pretrained-image-model \
    --text-model 'bert-base-uncased' --text-model-builder 'huggingface' --adapter 'lang_adapter' --text-head-n-layers 2 \
    --distiller 'InfoNCE' --lock-image-model \
    --report-to tensorboard --logs 'logs/BestPractice-Adapter' --cache-dir 'cache/weights' --name 'lang_adapter'
```
- **Single GPU, load pretrained image models**

```bash
python src/training/main.py \
    --dataset-size 2500000 --episode-size 2500000 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 0 --save-frequency 1 --batch-size 100 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --image-model 'dino_vits16' --image-model-tag 'facebookresearch/dino:main' --image-model-builder 'torchhub' \
    --unlock-image-model --unlock-text-model --text-model 'RN50' \
    --text-model-builder 'openclip' \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/codebase_sample'
```
 
        resnet50 = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
    
    --image-model 'resnet50' --image-model-tag 'facebookresearch/swav:main' --image-model-builder 'torchhub' \
    --image-model 'dino_vits16' --image-model-tag 'facebookresearch/dino:main' --image-model-builder 'torchhub' \
    --image-model 'resnet50' --image-model-tag 'facebookresearch/vicreg:main' --image-model-builder 'torchhub' \
    --image-model 'resnet50' --image-model-tag 'facebookresearch/barlowtwins:main' --image-model-builder 'torchhub' \
    --image-model 'regnety_16gf' --image-model-tag 'facebookresearch/swag' --image-model-builder 'torchhub' \



- Train a CLIP from scratch:
```bash

torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 2500000 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 32 --save-frequency 32 --batch-size 64 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-tag 'openai' --image-model-builder 'OpenCLIP' --image-head-n-layers 0 \
    --text-model 'RN50' --text-model-tag 'openai' --text-model-builder 'OpenCLIP' --unlock-text-model --text-head-n-layers 0 \
    --unlock-image-model  --unlock-text-model --cache-dir 'cache/weights' \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/codebase_test' --copy-codebase --name 'CLIP-from-scratch-bs512-CC2.5M-32ep'
```

- Train a CLIP from OpenAI-pretrained weights:
```bash
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 28 --save-frequency 28 --batch-size 100 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-tag 'openai' --image-model-builder 'OpenCLIP' --image-head-n-layers 0 \
    --text-model 'RN50' --text-model-tag 'openai' --text-model-builder 'OpenCLIP' --unlock-text-model --text-head-n-layers 0 \
    --unlock-image-model --pretrained-image-model --unlock-text-model --pretrained-text-model --cache-dir 'cache/weights' \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/codebase_test' --copy-codebase --name 'U[RN50-h0]-U[CLIP-from-RN50]-bs800-8ep'
```


- 加载预训练CLIP 的 text encoder, 从 torchvision 加载预训练的 MobileNet并冻结，进行 LiT
```bash
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src:src/training/evaluations"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 2000000 --train-data 'cache/yfcc_nori.csv' \
    --epochs 56 --save-frequency 56 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-5 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'mobilenet_v3_large'  --image-model-builder 'torchvision' --image-head-n-layers 2 \
    --text-model 'RN50' --text-model-tag 'openai' --text-model-builder 'openclip' --text-head-n-layers 0 \
    --pretrained-image-model  --cache-dir 'cache/weights' \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/codebase_test' --name 'L[mobilenet_v3_large-h2]-U[CLIP-from-scratch]-bs1024-YFCC-56ep-lr1e-5'
```
