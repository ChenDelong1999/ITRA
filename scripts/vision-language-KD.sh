
# Stage 2, Full fine-tuning (standard KD) with V100
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 500000 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 10 --save-frequency 10 --batch-size 128 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 0  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' --eval-first\
    --lr 1e-4 --warmup 100 --wd 0.5 --max-grad-norm 1 \
    --image-model 'ViT-B-16' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 1 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --unlock-text-model --text-model 'bert-base-uncased'  \
    --distiller 'InfoNCE' --find-unused-parameters \
    --report-to tensorboard --logs 'logs/V2L-KD' --copy-codebase --copy-codebase --name 'L[CLIP-ViT-B-16-h1]_[InfoNCE]_U[bert-base-uncased]-bs1024-10epX50w-CC'

# Stage 2, adaptive-tuning, V100, YFCC
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 500000 --train-data 'cache/yfcc_nori.csv' \
    --epochs 1000 --save-frequency 1000 --batch-size 256 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 0  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' --eval-first \
    --lr 5e-5 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --image-model 'ViT-B-16' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 1 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'bert-base-uncased'  \
    --distiller 'InfoNCE'  --adapter 'bottleneck_adapter' \
    --report-to tensorboard --logs 'logs/V2L-KD' --copy-codebase --name 'YFCC-L[CLIP-ViT-B-16-h1]_[InfoNCE]_L[bert-base-uncased]-bs2048-1000epX50w-bottleneck_adapter'

# Stage 2, adaptive-tuning
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 500000 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 300 --save-frequency 100 --batch-size 128 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 0  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' --eval-first \
    --lr 1e-4 --warmup 100 --wd 0.5 --max-grad-norm 1 \
    --image-model 'ViT-B-16' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 1 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'bert-base-uncased'  \
    --distiller 'InfoNCE' --adapter 'dummy' \
    --report-to tensorboard --logs 'logs/V2L-KD-0920' --copy-codebase --name 'L[CLIP-ViT-B-16-h1]_[InfoNCE]_L[bert-base-uncased]-bs1024-300epX50w-CC-dummy'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 

# datasets
    --dataset-size 14000000 --episode-size 500000 --train-data 'cache/yfcc_nori.csv' \
    --dataset-size 2500000 --episode-size 500000 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    # nori speedup 's3://chendelonghahab/datasets/ConceptualCaption3M/CC2.6M-CC2M.nori' --on --replica=2
    # nori speedup 's3://chendelong/datasets/ConceptualCaption3M/CC_3M.nori' --on --replica=2

# adapters
    --adapter 'prefix_tuning' --n-prompt 16
    --adapter 'bottleneck_adapter' 
    --adapter 'lang_adapter'
    --adapter 'dummy'
    --adapter 'mam_adapter'
    --text-head-n-layers 3 'adaption_head_3'

# teachers

    --image-model 'RN50' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 2 \
    --image-model 'ViT-B-16' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 2 \
    --image-model 'alexnet' --image-model-builder 'torchvision' --pretrained-image-model --image-head-n-layers 2 \alexnet
    --image-model 'resnet50' --image-model-builder 'torchvision' --pretrained-image-model --image-head-n-layers 2 \resnet50
        "AlexNet", 
            "alexnet"
        "VGG",
            "vgg11_bn",
            "vgg13_bn",
            "vgg16_bn",
            "vgg19_bn",
        "ResNet",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        "MobileNetV2", 
            "mobilenet_v3_small"
            "mobilenet_v3_large", 
        "VisionTransformer",
            "vit_b_32",
            "vit_b_16",
            "vit_l_32",
            "vit_l_16",

# students
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 
        --text-model 'bert-base-uncased' \
        --text-model 'bert-base-cased' \
        --text-model 'bert-large-uncased' \
        --text-model 'bert-large-cased' \
        --text-model 'roberta-base' \
        --text-model 'roberta-large' \
        --text-model 'roberta-large-mnli' \
        --text-model 'facebook/muppet-roberta-large' \
        --text-model 'Jean-Baptiste/roberta-large-ner-english' \
        --text-model 'princeton-nlp/unsup-simcse-roberta-large' \
        --text-model 'princeton-nlp/sup-simcse-roberta-large' \
        --text-model 'sentence-transformers/all-roberta-large-v1' \
        --text-model 'sentence-transformers/all-MiniLM-L12-v1' \
        --text-model 'sentence-transformers/msmarco-distilbert-base-tas-b' \
        --text-model 'xlm-roberta-large' \
        --text-model 'xlm-roberta-large-finetuned-conll03-english'\
        --text-model 'deepset/xlm-roberta-large-squad2' \
        --text-model 'joeddav/xlm-roberta-large-xnli' \
        --text-model 'facebook/contriever' \
        --text-model 'facebook/contriever-msmarco' \