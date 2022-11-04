# SimCSE Re-impl
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 1000000 --episode-size 100000 --train-data 'cache/simcse_wiki1m.csv' \
    --epochs 10 --save-frequency 10 --batch-size 64 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 0  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' --eval-first \
    --lr 1e-5 --warmup 0 --wd 0 --eps 1e-6 --max-grad-norm 1 \
    --image-model 'ViT-B-16' --image-model-builder 'OpenCLIP' --image-head-n-layers 0 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --unlock-text-model  --text-model 'roberta-base' \
    --text-pooler 'cls' --max-seq-length 32 --logit-scale 0.05 \
    --distiller 'InfoNCE' --w-simcse 1 --w-distill 0  \
    --report-to tensorboard --logs 'logs/V2L-KD-1027' --copy-codebase --name 'roberta_base-SimCSE-bs512-wiki1m-fix-temp'

# # PromptBERT Re-impl
# conda activate vlkd
# cd /data/codes/ProtoRKD
# export PYTHONPATH="src"
# eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
# ulimit -n 65536
# torchrun --nproc_per_node 8 -m training.main \
#     --dataset-size 1000000 --episode-size 100000 --train-data 'cache/simcse_wiki1m.csv' \
#     --epochs 10 --save-frequency 10 --batch-size 32 --workers 8 \
#     --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 0  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' --eval-first \
#     --lr 1e-5 --warmup 0 --wd 0 --eps 1e-6 --max-grad-norm 1 \
#     --image-model 'ViT-B-16' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 0 \
#     --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --unlock-text-model  --text-model 'roberta-base' \
#     --text-pooler 'PromptBERT' --max-seq-length 32 --logit-scale 0.05 \
#     --distiller 'InfoNCE' --w-simcse 1 --w-distill 0  \
#     --report-to tensorboard --logs 'logs/V2L-KD-1026-SimCSE-KD' --copy-codebase --name 'roberta_base-SimCSE(1.0)-KD(0.0)-bs256-wiki1m-PromptBERT-no-pooler'

# KD
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 100000 --train-data 'cache/yfcc_nori.csv' \
    --epochs 10 --save-frequency 10 --batch-size 64 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 0  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' --eval-first \
    --lr 1e-4 --warmup 100 --wd 0 --eps 1e-6 --max-grad-norm 1 \
    --image-model 'ViT-B-16' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 0 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 3 --unlock-text-model  --text-model 'roberta-base' \
    --text-pooler 'cls' --max-seq-length 32 --logit-scale 0.07 \
    --distiller 'CompRess-2q' --dino-teacher-temp 0.07 --teacher 'image' --w-simcse 0 --w-distill 1   \
    --report-to tensorboard --logs 'logs/V2L-KD-1101' --copy-codebase --name 'roberta_base-KD(CompRess-2q)-bs512-YFCC-1M-lr1e-4'


# KD-then-SimCSE
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 1000000 --episode-size 100000 --train-data 'cache/simcse_wiki1m.csv' \
    --epochs 20 --save-frequency 20 --batch-size 64 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 0  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' --eval-first \
    --lr 1e-5 --warmup 0 --wd 0 --eps 1e-6 --max-grad-norm 1 \
    --image-model 'ViT-B-16' --image-model-builder 'OpenCLIP' --image-head-n-layers 0 \
    --text-model-builder 'huggingface-transformer' --text-head-n-layers 0 --unlock-text-model  --text-model 'roberta-base' \
    --text-pooler 'cls' --max-seq-length 32 --logit-scale 0.05 \
    --distiller 'InfoNCE' --teacher 'image' --w-simcse 1 --w-distill 0  \
    --restart --resume 'logs/V2L-KD-1027/roberta_base-KD(InfoNCE)-bs512-YFCC-(x1)-h3:3/checkpoints/epoch_10.pt' \
    --report-to tensorboard --logs 'logs/V2L-KD-1027' --copy-codebase --name 'roberta_base-KD(x1)_then_SimCSE(x2)-wiki1m' --restart

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 

# datasets
    --dataset-size 1000000 --episode-size 100000 --train-data 'cache/simcse_wiki1m.csv' \
    --dataset-size 14000000 --episode-size 100000 --train-data 'cache/yfcc_nori.csv' \
    --dataset-size 2500000 --episode-size 100000 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
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