# **ProtoRKD: Prototypical Relational (Vision-Language) Knowledge Distillation**



# Requirements

## 1. Install Dependencies
- Create a conda environment and install PyTorch:

    ```bash
    conda create -n vlkd python=3.8
    conda activate vlkd
    ```

    This repo requirs PyTorch (1.11.0) and torchvision. Please install them via https://pytorch.org/get-started/locally

    ```
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch -y
    ```

<!-- - Clone this repo:

    ```bash
    git clone https://github.com/megvii-research/protoclip
    cd protoclip
    export PYTHONPATH="$PYTHONPATH:$PWD/src"
    ```
    **Note**: If import error is occured later, run `export PYTHONPATH="$PYTHONPATH:$PWD/src"` again. -->

- Install additional dependencies:
    ```bash
    conda install pillow pandas scikit-learn faiss-gpu ftfy tqdm matplotlib pycocotools 
    conda install -c huggingface transformers 
    conda install -c conda-forge sentence-transformers
    conda install wandb
    pip install adapter-transformers
    # TODO: remove nori dependency
    pip install nori2
    ```

    ```
    ELEVATOR:
    pip install yacs timm git+https://github.com/haotian-liu/CLIP_vlp.git vision-evaluation

    yacs~=0.1.8
    scikit-learn
    timm~=0.4.12
    numpy~=1.21.0
    sharedmem
    git+https://github.com/openai/CLIP.git
    git+https://github.com/haotian-liu/CLIP_vlp.git
    torch~=1.7.0
    PyYAML~=5.4.1
    Pillow~=9.0.1
    torchvision~=0.8.0
    vision-evaluation>=0.2.2
    vision-datasets>=0.2.0
    tqdm~=4.62.3
    transformers~=4.11.3
    protobuf~=3.20.1
    ftfy~=6.1.1
    nltk~=3.7

    ```
    

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

- *STS*
https://github.com/princeton-nlp/SimCSE#evaluation

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