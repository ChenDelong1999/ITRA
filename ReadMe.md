
<p align="center">
<img style="vertical-align:middle" src="assets/pipeline.png" />
</p>

<h1 align="center">
<span>[My Codebase Name]</span>
</h1>

<h2 align="center">
A flexible codebase for vision language learning.
</h2>


# Model Architechture

## **Image Backbone**
- From `OpenCLIP` (v2.0.2)

    - ResNet
    - Vision Transformer
    
    [OpenCLIP](https://github.com/mlfoundations/open_clip) is an open source implementation of [OpenAI's CLIP](https://github.com/openai/CLIP) (Contrastive Language-Image Pre-training). To check all supported model architecture and pretrained weigths, run:

    ```python
    >>> import open_clip
    >>> open_clip.list_pretrained()
    ```

- From `Torchvision` [(v0.12)](https://pytorch.org/vision/0.12/)
    - AlexNet
    - ResNet
    - Mobilenet
    - EfficientNet
    - DenseNet
    - ConvNext
    - Vision Transformer
    - ...

    To check all supported model architecture and pretrained weigths, run the following command or see [this page](https://pytorch.org/vision/0.12/models.html)

    ```python
    >>> import torchvision
    >>> torchvision.models.__dict__.keys()
    ```

## **Text Backbone**
- From `OpenCLIP`
    Choices of the text encoder is the same as OpenCLIP's image backbone.

- 🤗 From Hugging Face Transformer    
    - BERT (`bert-base-uncased`)
    - RoBERTa (`roberta-base-cased`)
    - ...

        For more details, see [Hugging Face Transformers](https://huggingface.co/docs/transformers). Currently, only 'from pretrained' mode is supported (i.e., you cannot train a huggingface transformer from scratch now). 
        
        Standard models like BERT/RoBERTa are supported, but whether other models are also supported is not sure...

    - [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html)
        - SBERT
        - Semantic Search Models
        - Word Embeddings (GloVe)

            Loading sentence transformers via huggingface and specify `--text-pooler='mean'` is recommended, though it is also supported to load the model via sentence transformer:

            ```bash
            # recommended: 
            --text-model-builder 'huggingface'  --text-model 'sentence-transformers/all-mpnet-base-v2' --text-pooler='mean' 
            # not recommended:
            --text-model-builder 'sbert'  --text-model 'all-mpnet-base-v2' 
            ```

    - Adapted Huggingface Transformer (via [Adapter-Transformers](https://github.com/adapter-hub/adapter-transformers))
        - [Bottleneck adapters](https://docs.adapterhub.ml/overview.html#bottleneck-adapters)
        - [Language Adapters](https://docs.adapterhub.ml/overview.html#language-adapters-invertible-adapters) 
        - [Prefix Tuning](https://docs.adapterhub.ml/overview.html#prefix-tuning)
        - [Compacter](https://docs.adapterhub.ml/overview.html#compacter)
        - [LoRA](https://docs.adapterhub.ml/overview.html#lora)
        - [(IA)^3](https://docs.adapterhub.ml/overview.html#ia-3)
        - [Mix-and-Match   Adapters](https://docs.adapterhub.ml/overview.html#mix-and-match-adapters)
        - [UniPELT](https://docs.adapterhub.ml/overview.html#unipelt)
        - ...

            [Adapter-Transformers](https://github.com/adapter-hub/adapter-transformers) is an extension of [HuggingFace's Transformers](https://github.com/huggingface/transformers) library, integrating adapters into state-of-the-art language models by incorporating [AdapterHub](https://adapterhub.ml/), a central repository for pre-trained adapter modules. 
            
            For more details, see: [Docs](https://docs.adapterhub.ml/) | [Model Overview](https://docs.adapterhub.ml/model_overview.html)

            | Method                                                                                                        | args.adapter       |         |
            |---------------------------------------------------------------------------------------------------------------|--------------------|------------|
            | [Bottleneck   adapters](https://docs.adapterhub.ml/overview.html#bottleneck-adapters)                         | `bottleneck_adapter` |          |
            | [Language Adapters](https://docs.adapterhub.ml/overview.html#language-adapters-invertible-adapters)           | `lang_adapter`       |          |
            | [Prefix   Tuning](https://docs.adapterhub.ml/overview.html#prefix-tuning)                                     | `prefix_tuning`      |          |
            | [Compacter](https://docs.adapterhub.ml/overview.html#compacter)                                               | `dummy`              |          |
            | [LoRA](https://docs.adapterhub.ml/overview.html#lora)                                                         | `lora_adapter`       |          |
            | [(IA)^3](https://docs.adapterhub.ml/overview.html#ia-3)                                                       | `ia3_adapter`        |          |
            | [Mix-and-Match   Adapters](https://docs.adapterhub.ml/overview.html#mix-and-match-adapters)                   | `mam_adapter`        |          |
            | [UniPELT](https://docs.adapterhub.ml/overview.html#unipelt)                                                   | `unipelt`            |          |


## **Projection Head**

- Linear projection head

- [DINO MLP Head](https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/vision_transformer.py#L257) (optionally with a prototype layer in the last)



# Loss Function
- InfoNCE
- ...

| Loss            | Paper | Uni-Directional | Need Prototype Layer |
|-----------------|-------|-----------------|----------------------|
| InfoNCE         | CLIP  |                 |                      |
| SimReg          |       |                 |                      |
| SimReg-L1       |       |                 |                      |
| SimReg-SmoothL1 |       |                 |                      |
| VICReg          |       |                 |                      |
| BarlowTwins     |       |                 |                      |
| RKD             |       |                 |                      |
| CompRess-1q     |       | &#10004;        |                      |
| CompRess-2q     |       |                 |                      |
| SEED            |       | &#10004;        |                      |
| DINO            |       | &#10004;        | &#10004;             |
| ProtoCPC        |       | &#10004;        | &#10004;             |


# Downstream Task
- Image Classification (ELEVATER?)
- Image-text Retrieval
- Sentence Similarity
- MS MARCO Passage Retrval...


# Sample Commands

- Train a CLIP from scratch:
```bash
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
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


- 加载 Hugging Face 预训练 BERT 并使用 Adapter, 从 torchvision 加载预训练的 MobileNet并冻结，进行 LiT
```bash
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 32 --save-frequency 32 --batch-size 100 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'mobilenet_v3_small'  --image-model-builder 'torchvision' --image-head-n-layers 2 \
    --text-model 'bert-base-uncased' --text-model-builder 'huggingface' --text-head-n-layers 1 \
    --lock-image-model --pretrained-image-model --lock-text-model --pretrained-text-model --adapter 'bottleneck_adapter' --cache-dir 'cache/weights' \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/codebase_test' --copy-codebase --name 'L[mobilenet_v3_small-h2]-L[bert-base-uncased-bottleneck_adapter]-bs800-8ep'
```