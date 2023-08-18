lg=$1
exp_name=tayi-CN
CUDA_VISIBLE_DEVICES=0 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate_taiyi.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 0 \
    --use_multi \
    --num_multi 8 &
CUDA_VISIBLE_DEVICES=1 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate_taiyi.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 1 \
    --use_multi \
    --num_multi 8 &
CUDA_VISIBLE_DEVICES=2 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate_taiyi.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 2 \
    --use_multi \
    --num_multi 8 &
CUDA_VISIBLE_DEVICES=3 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate_taiyi.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 3 \
    --use_multi \
    --num_multi 8 &
CUDA_VISIBLE_DEVICES=4 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate_taiyi.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 4 \
    --use_multi \
    --num_multi 8 &
CUDA_VISIBLE_DEVICES=5 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate_taiyi.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 5 \
    --use_multi \
    --num_multi 8 &
CUDA_VISIBLE_DEVICES=6 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate_taiyi.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 6 \
    --use_multi \
    --num_multi 8 &
CUDA_VISIBLE_DEVICES=7 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate_taiyi.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 7 \
    --use_multi \
    --num_multi 8 