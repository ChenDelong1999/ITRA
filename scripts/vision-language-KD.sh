
# Stage 2, Full fine-tuning (standard KD) with V100
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 2000000 --train-data 'cache/yfcc_nori.csv' \
    --epochs 56 --save-frequency 56 --batch-size 32 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --pretrained-image-model --image-head-n-layers 2 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --unlock-text-model --text-model 'roberta-large'  \
    --distiller 'InfoNCE' --find-unused-parameters \
    --report-to tensorboard --logs 'logs/vision-to-language-KD' --copy-codebase --name 'L[RN50-h2]_[InfoNCE]_U[all-roberta-large-v1]-bs256-8ep'



# Stage 2, adaptive-tuning
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 1 -m training.main \
    --dataset-size 14000000 --episode-size 2000000 --train-data 'cache/yfcc_nori.csv' \
    --epochs 56 --save-frequency 56 --batch-size 32 --workers 8 \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 1 \
    --image-model 'vit_l_32' --image-model-builder 'torchvision' --pretrained-image-model --image-head-n-layers 2 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'roberta-large'  \
    --distiller 'InfoNCE'  --adapter 'lang_adapter' \
    --report-to tensorboard --logs 'logs/vision-to-language-KD-teacher-ablation' --copy-codebase --name 'L[vit_b_16]_[InfoNCE]_L[all-roberta-large-v1]-bs256-8ep-lang_adapter'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

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

# adapters
    --adapter 'prefix_tuning' --n-prompt 16
    --adapter 'bottleneck_adapter' 
    --adapter 'lang_adapter'
    --adapter 'dummy'
    --adapter 'mam_adapter'
    --text-head-n-layers 3 'adaption_head_3'