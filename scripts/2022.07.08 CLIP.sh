


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Baselines # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# SimReg baseline.
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'SimReg' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[SimReg]_bs320_lr25e-6_16-epochs(CC250w)'

# CompRess-1q baseline.
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CompRess-1q' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[CompRess-1q]_bs320_lr25e-6_16-epochs(CC250w)'

# CompRess-2q baseline.
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CompRess' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[CompRess-2q]_bs320_lr25e-6_16-epochs(CC250w)'

# RKD baseline.
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'RKD' --w-rkd-d 1 --w-rkd-a 1 \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[RKD-DA]_bs320_lr25e-6_16-epochs(CC250w)'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # Baselines + proj. # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# SimReg w/ teacher proj. + projection AE
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-teacher-projection-head --res-teacher-projection --add-teacher-projection-AE  --w-projection-AE 10000 --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'SimReg' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2-res-proj.AE(1w)]_stu[RN50]_[SimReg]_bs320_lr25e-6_16-epochs(CC250w)'

# CompRess-1q w/ teacher proj. + projection AE
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-teacher-projection-head --res-teacher-projection --add-teacher-projection-AE --w-projection-AE 10000 --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CompRess-1q' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2-res-proj.AE(1w)]_stu[RN50]_[CompRess-1q]_bs320_lr25e-6_16-epochs(CC250w)'

# RKD w/ teacher proj. + projection AE
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-teacher-projection-head --res-teacher-projection --add-teacher-projection-AE  --w-projection-AE 10000 --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'RKD' --w-rkd-d 1 --w-rkd-a 0 \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2-res-proj.AE(1w)]_stu[RN50]_[RKD-D]_bs320_lr25e-6_16-epochs(CC250w)'

# RKD w/ teacher proj. + projection AE
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-teacher-projection-head --res-teacher-projection --add-teacher-projection-AE  --w-projection-AE 10000 --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'RKD' --w-rkd-d 1 --w-rkd-a 0 \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2-res-proj.AE(1w)]_stu[RN50]_[RKD-D]_bs320_lr25e-6_16-epochs(CC250w)'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # #    CLIP   # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# CLIP w/o teacher proj. Baseline
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[CLIP]_bs320_lr25e-6_16-epochs(CC250w)'

# CLIP w/o teacher proj. larger student
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN101 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN101]_[CLIP]_bs320_lr25e-6_16-epochs(CC250w)'

# CLIP w/o teacher proj. smaller student
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model alexnet --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[alexnet]_[CLIP]_bs320_lr25e-6_16-epochs(CC250w)'

# CLIP w/o teacher proj. CLIP teacher
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'clip-ViT-L-14' \
    --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[clip-ViT-L-14]_stu[RN50]_[CLIP]_bs320_lr25e-6_16-epochs(CC250w)'

# CLIP w/o teacher proj. YFCC data
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 14000000 --episode-size 2500000 --train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[CLIP]_bs320_lr25e-6_16-epochs(YFCC250w)'

# # #  + teacher projection
# CLIP w/ teacher proj.
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-teacher-projection-head --res-teacher-projection --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2-res-proj.]_stu[RN50]_[CLIP]_bs320_lr25e-6_16-epochs(CC250w)'

# CLIP w/ teacher proj.  + projection AE
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 4 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-teacher-projection-head --add-teacher-projection-AE --w-projection-AE 1000000 --res-teacher-projection --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2-res-proj.AE(100w)]_stu[RN50]_[CLIP]_bs320_lr25e-6_16-epochs(CC250w)'



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # #  CLIP  (8x2080ti) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# CLIP w/o teacher proj.
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 64 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2]_stu[RN50]_[CLIP]_bs640_lr1e-4_64-epochs(CC250w)'

# CLIP w/o teacher proj. larger student
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 40 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model ViT-B-16 --open-clip-model --pretrained-text 'clip-ViT-B-16' \
    --add-projection-head --projection-dim 512 --projection-n-layers 4 \
    --lr 25e-6 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[clip-ViT-B-16]_stu[ViT-B-16]_[CLIP]_bs320_lr25e-6_16-epochs(CC250w)'


# CLIP w/ teacher proj.  w/o projection AE
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-teacher-projection-head --res-teacher-projection --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2-res-proj]_stu[RN50]_[CLIP]_bs640_lr1e-4_32-epochs(CC250w)'


# CLIP w/ teacher proj.  + projection AE
conda activate protoclip
cd /data/codes/ProtoRKD
export PYTHONPATH="src"
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 2500000 --episode-size 0 --train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv' \
    --epochs 16 --save-frequency 4 --batch-size 80 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --eval-data-dir '/data/Datasets' \
    --model RN50 --open-clip-model --pretrained-text 'all-mpnet-base-v2' \
    --add-teacher-projection-head --add-teacher-projection-AE --w-projection-AE 10000 --res-teacher-projection --add-projection-head --projection-dim 768 --projection-n-layers 4 \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 10 \
    --distiller 'CLIP' \
    --report-to tensorboard --logs logs/exp6_clip_teacher_proj --copy-codebase --name 'tea[all-mpnet-base-v2-res-proj.AE(1w)]_stu[RN50]_[CLIP]_bs640_lr1e-4_32-epochs(CC250w)'
