
# CLIP from scratch
conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 4000000 --train-data 'cache/yfcc_nori.csv' \
    --epochs 28 --save-frequency 28 --batch-size 100 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --unlock-image-model --image-head-n-layers 2 \
    --text-model 'RN50' --unlock-text-model --text-head-n-layers 0 \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/Stage1-teacher-ablation' --copy-codebase --name 'U[RN50-h2]-U[CLIP-from-scratch]-bs800-8ep'

# # pretrained but unlock
# conda activate vlkd
# cd /data/codes/ProtoRKD
# export PYTHONPATH="src"
# eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
# ulimit -n 65536
# torchrun --nproc_per_node 8 -m training.main \
#     --dataset-size 14000000 --episode-size 4000000 --train-data 'cache/yfcc_nori.csv' \
#     --epochs 28 --save-frequency 28 --batch-size 100 --workers 8 \
#     --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
#     --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
#     --image-model 'RN50' --image-model-builder 'OpenCLIP' --unlock-image-model --image-head-n-layers 2 \
#     --text-model-builder 'huggingface-transformer' --unlock-text-model --pretrained-text-model --text-head-n-layers 0 --text-model 'sentence-transformers/all-roberta-large-v1' \
#     --distiller 'InfoNCE' --find-unused-parameters \
#     --report-to tensorboard --logs 'logs/Stage1-teacher-ablation' --copy-codebase --name 'U[RN50-h2]-U[all-roberta-large-v1]-bs800-8ep'


# teacher ablation  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 8 epoch (4M x 28) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

conda activate vlkd
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 4000000 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 28 --save-frequency 28 --batch-size 100 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --unlock-image-model --image-head-n-layers 2 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'facebook/contriever' \
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/Stage1-teacher-ablation' --copy-codebase --name 'U[RN50-h2]-L[facebook-contriever]-bs800-8ep'
    
# Teacher Zoo
    --text-model 'RN50' --text-model-builder 'OpenCLIP' --pretrained-text-model --text-head-n-layers 0 \CLIP-pretrained
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
    --text-model-builder 'sentence-transformer' --pretrained-text-model --text-head-n-layers 0 
        --text-model 'all-mpnet-base-v2' \
        --text-model 'average_word_embeddings_komninos' \
        --text-model 'average_word_embeddings_glove.6B.300d' \


# distiller ablation  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 4 epoch (2M x 28) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
ulimit -n 65536
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 2000000 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 28 --save-frequency 28 --batch-size 64 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'OpenCLIP' --unlock-image-model --image-head-n-layers 2 \
    --text-model-builder 'huggingface-transformer' --pretrained-text-model --text-head-n-layers 0 --text-model 'sentence-transformers/all-roberta-large-v1' \
    --distiller 'ProtoCPC' \
    --report-to tensorboard --logs 'logs/Stage1-distiller-ablation' --copy-codebase --name 'U[RN50-h2]-[ProtoCPC-copy]-L[sentence-transformers-all-roberta-large-v1]-bs512-8ep'


# Distillers Zoo
    --distiller 'SimReg' \
    --distiller 'SimReg-L1' \
    --distiller 'SimReg-SmoothL1' \
    --distiller 'VICReg' \
    --distiller 'BarlowTwins' \
    --distiller 'RKD' --w-rkd-d 1 --w-rkd-a 0 \
    --distiller 'RKD' --w-rkd-d 0 --w-rkd-a 1 \ (angle loss CUDA OOM) 
    --distiller 'RKD' --w-rkd-d 1 --w-rkd-a 1 \ (angle loss CUDA OOM) 
    --distiller 'InfoNCE' \
    --distiller 'CompRess-1q' \
    --distiller 'CompRess-2q' \
    --distiller 'SEED' \
    --distiller 'DINO' \
    --distiller 'ProtoCPC' \
    