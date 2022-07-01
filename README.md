# **ProtoRKD: Prototypical Relational (Vision-Language) Knowledge Distillation**


## Single GPU Training Script
```bash
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
python src/training/main.py \
    --dataset-size 2500000 --episode-size 250000 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 4 --save-frequency 1 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 300 --projection-n-layers 2 --pretrained-text 'average_word_embeddings_glove.6B.300d' \
    --lr 5e-5 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[average_word_embeddings_glove.6B.300d]_stu[RN50]_[MSE]_bs128_lr5e-5_4-epochs(CC250w)'
```

**Note**: 尽量不要在训练过程中进行评测：在多卡环境下这可能会导致pytorch dataloader死锁。相反，建议在训练完成后使用`python src/utils/evaluate_checkpoints.py`进行批量评测。

## Multiple GPU Training Script
```bash
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 128 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --add-projection-head --projection-dim 300 --projection-n-layers 2 --pretrained-text 'average_word_embeddings_glove.6B.300d' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --report-to tensorboard --logs logs --copy-codebase --name 'tea[average_word_embeddings_glove.6B.300d]_stu[RN50]_[MSE]_bs1024_lr5e-4_16-epochs(CC250w)'
```


# Requirements

## 1. Install Dependencies
- Create a conda environment and install PyTorch:

    ```bash
    conda create -n protoclip python=3.8
    conda activate protoclip
    ```

    This repo requirs PyTorch (1.11.0) and torchvision. Please install them via https://pytorch.org/get-started/locally

- Clone this repo:

    ```bash
    git clone https://github.com/megvii-research/protoclip
    cd protoclip
    export PYTHONPATH="$PYTHONPATH:$PWD/src"
    ```
    **Note**: If import error is occured later, run `export PYTHONPATH="$PYTHONPATH:$PWD/src"` again.

- Install additional dependencies:
    ```bash
    conda install pandas scikit-learn faiss-gpu ftfy tqdm matplotlib pycocotools
    pip install pytorch-transformers
    conda install -c huggingface transformers # hugging face
    conda install wandb # if you want to use wandb for better logging
    # TODO: remove nori dependency
    pip install nori2
    ```
    **Note**: This codebase integrate [pytorch-transformers](https://pypi.org/project/pytorch-transformers) to initalize the text tower with large pretrained language model (experimental).


## 2. Prepare Pretraining Data
This codebase reads a `CSV` file (separated by `\t`) with two columns: a path to an image ("filepath" by default), and a text caption ("title" by default).

| filepath          | title                      |
|-------------------|----------------------------|
| path/to/image.jpg | A very typical bus station |
| ...               | ...                        |

The script `src/utils/gather_cc.py` will collect the [Conceptual Captions](https://github.com/google-research-datasets/conceptual-captions) (CC3M) dataset. First, download the Conceptual Captions URLs from [here](https://ai.google.com/research/ConceptualCaptions/download), then run the following script:

```bash
python3 src/utils/gather_cc.py path/to/Train_GCC-training.tsv
```

**Note**: The requirement of CC3M validation data of OpenCLIP is removed in this codebase. The CC3M dataset was made public by Google in 2018. As noted in our paper, the number of accessible images keeps drooping due to expired image links. This issue is raised by several recent works. In this work, since we can only collect 2,643,718 images (concurrent to our work, [CyCLIP](https://arxiv.org/abs/2205.14459) collected 2,631,703 images), we randomly sample a 2,500,000 subset (75\% of full CC3M) from them to train our ProtoCLIP. Considering the dropping accessibility of image links in Conceptual Captions, we call for the use of this dataset size (2.5M) in future benchmarking for better comparability.

**Note**: `webdataset` is no longer supported in this codebase.


## 3. Prepare Downstream Data
- **Zero-shot Classification**. The preprocessed zero-shot datasets can be downloaded from [CLOOB](https://github.com/ml-jku/cloob#downstream-tasks).

- **Linear Probing**. We perform linear evaluation on ImageNet, CIFAR10, CIFAR100, and STL10. You need to download the full [ImageNet-1k](https://image-net.org/download.php) dataset manually. The later three datasets are integrated into `torchvision` and will be downloaded automatically.

- **Image-text Retrieval**. We implement zero-shot image-text retrieval on MS-COCO. Since we do not perform fine-tuning, only the validation split (`/val2017`) is required here.

    
    ```
    # All downstream datasets shall be stored to <YOUR DATASET ROOT> dictionary:
    <YOUR DATASET ROOT>
        ├── imagenet
        │   ├── train
        │   └── test  
        ├── birdsnap
        │   └── test
        ├── country211
        │   └── test
        ├── flowers102
        │   └── test
        ├── gtsrb
        │   └── test
        ├── stanford_cars
        │   └── test
        ├── ucf101
        │   ├── testlist01
        │   ├── testlist02
        │   └── testlist03   
        └── coco2017
           ├── annotations
           └── val2017 
    ```


## 📈Monitoring Downstream Performances During Training

Experiment will be logged to `<Your Experiment Log dir>` as following:
```
<Your Experiment Log dir>
    ├── cache
    ├── checkpoints
    │   ├── epoch_4.pt
    │   ├── epoch_8.pt
    │   ├── epoch_12.pt
    │   ├── epoch_16.pt
    │   ├── epoch_20.pt
    │   ├── epoch_24.pt
    │   ├── epoch_28.pt
    │   ├── epoch_32.pt
    │   └── epoch_latest.pt
    ├── out.log
    ├── params.txt
    ├── results.jsonl
    ├── evaluation_metrics_all.csv
    └── tensorboard
        └── events.out.tfevents
```

We present an useful tool for monitoring the downstream performance. By running `src/utils/evaluate_checkpoints.py` and specifying an experiment logging dir, it will read configurations from `params.txt` and automatically monitor and evaluate checkpoints. The result will be automatically saved as a `.csv` file (`evaluation_metrics_all.csv`). You can also specify an individual checkpoint to evaluate.
```
>>> python src/utils/evaluate_checkpoints.py
Please input your experiment dir: <Your Experiment Log dir>
Specify a checkpoint epoch? (press "enter" to scan and evaluate all checkpoints) 
```