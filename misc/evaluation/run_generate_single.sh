CUDA_VISIBLE_DEVICES=0 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/generate.py \
    --exp_name laion5plus_256_90k \
    --rank 0 \
    --ckpt /share/project/yfl/codebase/git/AltTools/Altdiffusion/ckpt/laion5plus_256_kv/checkpoints/step=000090000.ckpt &
CUDA_VISIBLE_DEVICES=1 python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/generate.py \
    --exp_name laion5plus_256_60k \
    --rank 1 \
    --ckpt /share/project/yfl/codebase/git/AltTools/Altdiffusion/ckpt/laion5plus_256_kv/checkpoints/step=000060000.ckpt