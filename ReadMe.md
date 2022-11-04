
This repository supports the following libararies:

- [OpenCLIP](https://github.com/mlfoundations/open_clip): An open source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training).

    pip install open-clip-torch==2.0.2


# Image Encoder

Tips: 

在通过 OpenCLIP 加载预训练权重之前，如果您使用 torchrun 进行多卡训练，请首先运行如下命令：

```bash
python
>>> import open_clip
>>> open_clip.create_model_and_transforms(model_name='RN50', pretrained='openai', cache_dir='cache/weights/open_clip')
```


# Sample Commands

- Train a CLIP from scratch:
```
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 32 --save-frequency 32 --batch-size 64 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-tag 'openai' --image-model-builder 'OpenCLIP' --image-head-n-layers 0 \
    --text-model 'RN50' --text-model-tag 'openai' --text-model-builder 'OpenCLIP' --unlock-text-model --text-head-n-layers 0 \
    --unlock-image-model  --unlock-text-model --cache-dir 'cache/weights' \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/codebase_test' --copy-codebase --name 'CLIP-from-scratch-bs512-32ep'
```

- Train a CLIP from OpenAI-pretrained weights:
```
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