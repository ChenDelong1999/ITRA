
<p align="center">
<img style="vertical-align:middle" src="assets/pipeline.png" />
</p>

<h3 align="center">
<span>[My Codebase Name]</span>
</h3>

<h4 align="center">
A compositional codebase for flexible vision language learning.
</h4>


**Model Builder**
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Torchvision (v0.12)](https://pytorch.org/vision/0.12/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html)
- [Adapter-Transformers](https://github.com/adapter-hub/adapter-transformers)

**Training Objectives**
- CLIP: InfoNCE, ProtoCLIP
- Self-supervised KD: RKD, SEED, CompRess, ProtoCPC, SimReg
- VICReg, BarlowTwins, DINO

**Downstream Evaluation**
- Image classification: zero-shot, linear/k-NN, and clustering evaluation (AMI, NMI) (from [ProtoCLIP](https://github.com/megvii-research/protoclip))
- [EVEVATER Image Classification Toolkit](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC) on 20 datasets
- Image-text retrieval on MS-COCO dataset
- Sentence embeddings ([SentEval](https://github.com/facebookresearch/SentEval))
- Passage retrieval on MS-MARCO and Wiki Sections
- Word embeddings: RG65, Simlex999, WordSim353




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

    Here the alternatives of the text encoder are exactly the same as OpenCLIP's image backbone.

- 🤗 From Hugging Face Transformer    
    - BERT (`bert-base-uncased`)
    - RoBERTa (`roberta-base-cased`)
    - ...

        For more details, see [HuggingFace Transformers](https://huggingface.co/docs/transformers). Currently, only 'from pretrained' mode is supported (i.e., you cannot train a huggingface transformer from scratch now). 
        
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
    --pretrained-image-model --lock-text-model --pretrained-text-model --cache-dir 'cache/weights' \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/codebase_test' --name 'L[mobilenet_v3_large-h2]-L[CLIP-from-RN50]-bs1024-YFCC-8ep-lr1e-5'
```

# Evaluation

## EVEVATER Image Classification Toolkit

[EVEVATER Image Classification Toolkit](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC) (Elevater_Toolkit_IC) implemeted standarlized evaluations of vision language models. It covers zero-shot classification, few- / full-shot linear probing, and fully fine tuning on 20 datasets. See paper "*[ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models](https://arxiv.org/abs/2204.08790), NeurIPS 2022 Datasets and Benchmarks Track*" for more details.

We have included Elevater_Toolkit_IC in our codebase (in `src/training/evaluations/vision_benchmark`). We have registered new models ([clip_zeroshot_eval.py]((src/training/evaluations/vision_benchmark/models/clip_zeroshot_eval.py)) and [cls_linear_or_ft_eval.py]((src/training/evaluations/vision_benchmark/models/cls_linear_or_ft_eval.py))) following the official instructions. To ensure compatibility, we have made some modifications based on the official Elevater_Toolkit_IC codes at commit `9d39620`, so DO NOT install an Elevater_Toolkit_IC in the environment for this codebase.

To get started first download all dataset following [this repo](https://github.com/Computer-Vision-in-the-Wild/DataDownload). The downloaded datasets takes about 41Gb storage, and the folder structure should be: 


```bash
.../datasets
└── classification
    ├── caltech_101_20211007
    │   ├── labels.txt
    │   ├── test.txt
    │   ├── test.zip
    │   ├── train.txt
    │   └── train.zip
    ├── cifar100_20200721
    │   ├── labels.txt
    │   ├── test_images.txt
    │   ├── test_images.zip
    │   ├── train_images.txt
    │   └── train_images.zip
    ...
    └── voc2007_20211007
        ├── labels.txt
        ├── test_ic.txt
        ├── test.zip
        ├── train_ic.txt
        ├── train.zip
        └── val_ic.txt

21 directories, 115 files
```

Then you can perform EVEVATOR evaluations of the model trained by this codebase, by making necessary modifications and run the following commands:

```bash
conda activate vlkd
cd /data/codes/ProtoRKD 
export PYTHONPATH="$PWD/src/training/evaluations:$PWD/src"

# zero-shot:       model_cfg='clip_zeroshot_eval'      mode='zeroshot'\
# few-shot:        model_cfg='cls_linear_or_ft_eval'   mode='linear_probe' num_shots=5 \
# linear prob:     model_cfg='cls_linear_or_ft_eval'   mode='linear_probe' num_shots=-1 \
# fine-tune:       model_cfg='cls_linear_or_ft_eval'   mode='finetune'     num_shots=-1 \

for dataset (caltech101 cifar10 cifar100 country211 dtd eurosat-clip fer2013 fgvc-aircraft-2013b flower102 food101 gtsrb hateful-memes kitti-distance mnist oxford-iiit-pets patchcamelyon rendered-sst2 resisc45-clip stanfordcar voc2007classification)
{       
    #---> REPLACE THIS LINE WITH ONE OF FOUR OPTIONS ABOVE <---#
    log_dir=# <YOUR EXPERIMENT DIR> \
    ckpt_epoch=# <WHICH EPOCH> \
    dataset_root=# <YOUR DATASET DIR> \
    dataset=$dataset \
    disable_hyperparameter_tuning=True \
        bash run_evevater_eval.sh
}
```

for example,
```base
conda activate vlkd
cd /data/codes/ProtoRKD 
export PYTHONPATH="$PWD/src/training/evaluations:$PWD/src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)

for dataset (caltech101 cifar10 cifar100 country211 dtd eurosat-clip fer2013 fgvc-aircraft-2013b flower102 food101 gtsrb hateful-memes kitti-distance mnist oxford-iiit-pets patchcamelyon rendered-sst2 resisc45-clip stanfordcar voc2007classification)
{ 
    model_cfg='clip_zeroshot_eval'      mode='zeroshot'\
    log_dir='/data/codes/ProtoRKD/logs/codebase_test/L[mobilenet_v3_large-h2]-L[CLIP-from-RN50]-bs1024-YFCC-56ep-lr1e-5' \
    ckpt_epoch=56 \
    dataset=$dataset \
    disable_hyperparameter_tuning=True \
    dataset_root='/data/codes/ProtoRKD/src/training/evaluations/vision_benchmark/outputs/datasets'\
        bash run_evevater_eval.sh
}

```

Then you can generate submission file for [EvalAI](https://eval.ai/web/challenges/challenge-page/1832/overview). For more details, please see [official instructions](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC#submit-to-leaderboard).


```bash
python src/training/evaluations/vision_benchmark/commands/prepare_submit.py \
  --combine_path 'logs/codebase_test/L[mobilenet_v3_small-h2]-L[CLIP-from-RN50]-bs1024-YFCC-8ep/clip_zeroshot_eval/log/predictions/zeroshot_eval_wiki_False_wnh_False_wnd_False_gpt3_Falseagg_WIKI_AND_GPT3_gpt3count_0'
```