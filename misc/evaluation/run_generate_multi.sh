lg=$1
ckpt=/share/project/yfl/codebase/git/AltTools/Altdiffusion/ckpt/xformer_laion5plus_512_kv/checkpoints/step=000005000.ckpt
# /share/project/yfl/codebase/git/AltTools/Altdiffusion/ckpt/xformer_laion5plus_512_kv_cfg/checkpoints/step=000015000.ckpt
# /share/project/yfl/codebase/git/AltTools/Altdiffusion/ckpt/ckpt_20230316/laion5b_128/checkpoints/step=000090000.ckpt
# /share/project/yfl/codebase/git/AltTools/Altdiffusion/ckpt/ckpt_20230316/step=000015000.ckpt
# /share/project/yfl/codebase/git/AltTools/Altdiffusion/ckpt/xformer_laion5plus_512_kv/checkpoints/step=000005000.ckpt
exp_name=laion5plus_512_kv_5k
CUDA_VISIBLE_DEVICES=0 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/generate.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 0 \
    --use_multi \
    --num_multi 8 \
    --ckpt $ckpt &
CUDA_VISIBLE_DEVICES=1 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/generate.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 1 \
    --use_multi \
    --num_multi 8 \
    --ckpt $ckpt &
CUDA_VISIBLE_DEVICES=2 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/generate.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 2 \
    --use_multi \
    --num_multi 8 \
    --ckpt $ckpt &
CUDA_VISIBLE_DEVICES=3 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/generate.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 3 \
    --use_multi \
    --num_multi 8 \
    --ckpt $ckpt &
CUDA_VISIBLE_DEVICES=4 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/generate.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 4 \
    --use_multi \
    --num_multi 8 \
    --ckpt $ckpt &
CUDA_VISIBLE_DEVICES=5 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/generate.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 5 \
    --use_multi \
    --num_multi 8 \
    --ckpt $ckpt &
CUDA_VISIBLE_DEVICES=6 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/generate.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 6 \
    --use_multi \
    --num_multi 8 \
    --ckpt $ckpt &
CUDA_VISIBLE_DEVICES=7 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/generate.py \
    --exp_name $exp_name \
    --lg $lg \
    --rank 7 \
    --use_multi \
    --num_multi 8 \
    --ckpt $ckpt