lg=$1
scale=$2
ckpt=/share/project/yfl/codebase/git/AltTools/Altdiffusion/ckpt/mv_back/step=000030000.ckpt
exp_name=stability_continue_train_kv
CUDA_VISIBLE_DEVICES=0 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 0 \
    --scale $scale \
    --ckpt $ckpt
    # --use_multi \
    # --num_multi 1 \
    # --use_ema \
# CUDA_VISIBLE_DEVICES=1 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate.py \
#     --exp_name $exp_name \
#     --lg $lg \
#     --rank 1 \
#     --use_multi \
#     --num_multi 8 \
#     --use_ema \
#     --scale $scale \
#     --ckpt $ckpt &
# CUDA_VISIBLE_DEVICES=2 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate.py \
#     --exp_name $exp_name \
#     --lg $lg \
#     --rank 2 \
#     --use_multi \
#     --num_multi 8 \
#     --use_ema \
#     --scale $scale \
#     --ckpt $ckpt &
# CUDA_VISIBLE_DEVICES=3 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate.py \
#     --exp_name $exp_name \
#     --lg $lg \
#     --rank 3 \
#     --use_multi \
#     --num_multi 8 \
#     --use_ema \
#     --scale $scale \
#     --ckpt $ckpt &
# CUDA_VISIBLE_DEVICES=4 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate.py \
#     --exp_name $exp_name \
#     --lg $lg \
#     --rank 4 \
#     --use_multi \
#     --num_multi 8 \
#     --use_ema \
#     --scale $scale \
#     --ckpt $ckpt &
# CUDA_VISIBLE_DEVICES=5 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate.py \
#     --exp_name $exp_name \
#     --lg $lg \
#     --rank 5 \
#     --use_multi \
#     --num_multi 8 \
#     --use_ema \
#     --scale $scale \
#     --ckpt $ckpt &
# CUDA_VISIBLE_DEVICES=6 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate.py \
#     --exp_name $exp_name \
#     --lg $lg \
#     --rank 6 \
#     --use_multi \
#     --num_multi 8 \
#     --use_ema \
#     --scale $scale \
#     --ckpt $ckpt &
# CUDA_VISIBLE_DEVICES=7 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate.py \
#     --exp_name $exp_name \
#     --lg $lg \
#     --rank 7 \
#     --use_multi \
#     --num_multi 8 \
#     --use_ema \
#     --scale $scale \
#     --ckpt $ckpt