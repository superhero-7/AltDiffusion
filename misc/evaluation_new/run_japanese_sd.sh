languages=("ja")

declare -A gpu_map
gpu_map=( ["ja"]=2)

# 启动所有后台任务
for lg in "${languages[@]}"
do
    gpu=${gpu_map[$lg]}
    CUDA_VISIBLE_DEVICES=$gpu python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/generate_ja.py \
        --lg $lg \
        --input_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/wit_${lg}.json \
        --save_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/ja_sd_generated_imgs/wit_${lg} \
        --model_base /share/project/yfl/database/hub/models--rinna--japanese-stable-diffusion/snapshots/07655518e5518c6ad4340168d1d0c98958e96ae0
done