# Adapters

conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="$PYTHONPATH:$PWD/src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --epochs 28 --save-frequency 28 --batch-size 32 \
    --dataset-size 1000000 --episode-size 500000 --train-data 'cache/yfcc_nori.csv'  --nori-dataset \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --pretrained-image-model --image-model-builder 'openclip'   --image-model 'RN50' --image-model-tag 'openai' --image-head-n-layers 0 \
    --pretrained-text-model  --text-model-builder 'huggingface' --text-model 'bert-base-uncased' --adapter 'prefix_tuning' --text-head-n-layers 2 \
    --distiller 'InfoNCE' --lock-image-model \
    --report-to tensorboard --logs 'logs/BestPractice-Adapter' --cache-dir 'cache/weights' --name '[openai-RN50]->[bert-base-uncased-2_proj]+prefix_tuning-YFCC1M_1epoch-lr5e-4-bs32'

# NO ADAPTERS
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="$PYTHONPATH:$PWD/src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --epochs 28 --save-frequency 28 --batch-size 32 \
    --dataset-size 1000000 --episode-size 500000 --train-data 'cache/yfcc_nori.csv' --nori-dataset \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --pretrained-image-model --image-model-builder 'openclip'   --image-model 'RN50' --image-model-tag 'openai' --image-head-n-layers 0 \
    --pretrained-text-model  --text-model-builder 'huggingface' --text-model 'bert-base-uncased' --text-head-n-layers 2 \
    --distiller 'InfoNCE' --lock-image-model \
    --report-to tensorboard --logs 'logs/BestPractice-Adapter' --cache-dir 'cache/weights' --name '[openai-RN50]->[bert-base-uncased-2_proj]-YFCC1M_1epoch-lr1e-4-bs32'

--adapter 'bottleneck_adapter'
--adapter 'lang_adapter'
--adapter 'prefix_tuning'
--adapter 'dummy'
--adapter 'lora_adapter'
--adapter 'ia3_adapter'
--adapter 'mam_adapter'
--adapter 'unipelt'
--adapter no adapters


# Image Pretrained Models
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="$PYTHONPATH:$PWD/src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --epochs 28 --save-frequency 28 --batch-size 32 \
    --dataset-size 14000000 --episode-size 500000 --train-data 'cache/yfcc_nori.csv'  --nori-dataset \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --pretrained-image-model --image-model-builder 'torchhub' --image-model 'resnet50_alpha0p9' --image-model-tag 'facebookresearch/vicregl:main' --image-head-n-layers 0 \
    --pretrained-text-model  --text-model-builder 'huggingface' --text-model 'bert-base-uncased' --adapter 'mam_adapter' --text-head-n-layers 2 \
    --distiller 'InfoNCE' --lock-image-model \
    --report-to tensorboard --logs 'logs/BestPractice-ImagePretrain' --cache-dir 'cache/weights' --name '[vicregl-resnet50_alpha0p9]->[bert-base-uncased-2_proj]+mam_adapter-YFCC_1epoch-lr5e-4-bs32'



- Smaller Models (<5M)
    MNASNet0_5_Weights.IMAGENET1K_V1            2.2M
    MNASNet0_75_Weights.IMAGENET1K_V1           3.2M
    MNASNet1_0_Weights.IMAGENET1K_V1            4.4M 
    MobileNet_V2_Weights.IMAGENET1K_V1          3.5M
    MobileNet_V2_Weights.IMAGENET1K_V2          3.5M
    MobileNet_V3_Small_Weights.IMAGENET1K_V1    2.5M
    RegNet_Y_400MF_Weights.IMAGENET1K_V1        4.3M
    RegNet_Y_400MF_Weights.IMAGENET1K_V2        4.3M
    ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1    1.4M
    ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1    2.3M
    ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1    3.5M
    SqueezeNet1_0_Weights.IMAGENET1K_V1         1.2M
    SqueezeNet1_1_Weights.IMAGENET1K_V1         1.2M

- ResNet50 (25.6M)
    --image-model-builder 'openclip' --image-model 'RN50' --image-model-tag 'openclip' --pretrained-image-model \
    --image-model-builder 'openclip' --image-model 'RN50' --image-model-tag 'yfcc15m' --pretrained-image-model  \
    --image-model-builder 'openclip' --image-model 'RN50' --image-model-tag 'cc12m' --pretrained-image-model  \
    --image-model-builder 'torchvision' --image-model 'resnet50' --pretrained-image-model \
    --image-model-builder 'torchhub' --image-model 'resnet50' --image-model-tag 'facebookresearch/swav:main' \
    --image-model-builder 'torchhub' --image-model 'resnet50' --image-model-tag 'facebookresearch/vicreg:main' \
    --image-model-builder 'torchhub' --image-model 'resnet50' --image-model-tag 'facebookresearch/barlowtwins:main' \
    --image-model-builder 'torchhub' --image-model 'resnet50_alpha0p9' --image-model-tag 'facebookresearch/vicregl:main' \
    --image-model-builder 'torchhub' --image-model 'resnet50_alpha0p75' --image-model-tag 'facebookresearch/vicregl:main' \

- ViT-B-16 (86.6M)
    --image-model-builder 'openclip' --image-model 'ViT-B-16' --image-model-tag 'openai' --pretrained-image-model  \
    --image-model-builder 'openclip' --image-model 'ViT-B-16' --image-model-tag 'cc12m' --pretrained-image-model  \
    --image-model-builder 'torchvision' --image-model 'vit_b_16' --pretrained-image-model \
    --image-model-builder 'torchhub' --image-model 'vit_b16' --image-model-tag 'facebookresearch/swag:main' \
    --image-model-builder 'torchhub' --image-model 'vit_b16_in1k' --image-model-tag 'facebookresearch/swag:main' \
 
- Larger Models (>200M)
    ConvNeXt_Large_Weights.IMAGENET1K_V 197.8M
    RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1 644.8M
    ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1 633.5M
    --image-model-builder 'openclip' --image-model 'ViT-H-14' --image-model-tag 'openai' --pretrained-image-model  \



conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="$PYTHONPATH:$PWD/src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --epochs 28 --save-frequency 28 --batch-size 32 \
    --dataset-size 14000000 --episode-size 500000 --train-data 'cache/yfcc_nori.csv' \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --pretrained-image-model --image-model-builder 'openclip'   --image-model 'RN50' --image-model-tag 'openai' --image-head-n-layers 0 \
    --pretrained-text-model  --text-model-builder 'huggingface' --text-model 'bert-base-uncased' --adapter 'lang_adapter' --text-head-n-layers 2 \
    --distiller 'InfoNCE' --lock-image-model \
    --report-to tensorboard --logs 'logs/BestPractice-Adapter' --cache-dir 'cache/weights' --name '[openai-RN50]->[bert-base-uncased-2_proj]+lang_adapter-YFCC_1epoch-lr5e-4-bs32'

