languages=("uk" "pl" "ko" "pt")

declare -A gpu_map
gpu_map=( ["uk"]=0 ["pl"]=1 ["ko"]=2 ["pt"]=3)

# 启动所有后台任务
for lg in "${languages[@]}"
do
    gpu=${gpu_map[$lg]}
    CUDA_VISIBLE_DEVICES=$gpu python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/stable_diffusion_api.py \
        --lg $lg \
        --input_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/wit_${lg}.json \
        --save_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/sd_generated_imgs/wit_${lg} \
        --model_base /share/project/yfl/database/ckpt/yfl/stable_diffusion_v2-1-base &
done

wait

languages=("vi")

declare -A gpu_map
gpu_map=(["vi"]=0)

# 启动所有后台任务
for lg in "${languages[@]}"
do
    gpu=${gpu_map[$lg]}
    CUDA_VISIBLE_DEVICES=$gpu python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/stable_diffusion_api.py \
        --lg $lg \
        --input_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/wit_${lg}.json \
        --save_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/sd_generated_imgs/wit_${lg} \
        --model_base /share/project/yfl/database/ckpt/yfl/stable_diffusion_v2-1-base &
done

# wait

# languages=("it" "ja" "nl" "ru")

# declare -A gpu_map
# gpu_map=(["it"]=0 ["ja"]=1 ["nl"]=2 ["ru"]=3)

# # 启动所有后台任务
# for lg in "${languages[@]}"
# do
#     gpu=${gpu_map[$lg]}
#     CUDA_VISIBLE_DEVICES=$gpu python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/stable_diffusion_api.py \
#         --lg $lg \
#         --input_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/wit_${lg}.json \
#         --save_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/sd_generated_imgs/wit_${lg} \
#         --model_base /share/project/yfl/database/ckpt/yfl/stable_diffusion_v2-1-base &
# done

# wait

# languages=("th" "tr" "zh")

# declare -A gpu_map
# gpu_map=(["th"]=0 ["tr"]=1 ["zh"]=2)

# # 启动所有后台任务
# for lg in "${languages[@]}"
# do
#     gpu=${gpu_map[$lg]}
#     CUDA_VISIBLE_DEVICES=$gpu python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/stable_diffusion_api.py \
#         --lg $lg \
#         --input_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/wit_${lg}.json \
#         --save_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/sd_generated_imgs/wit_${lg} \
#         --model_base /share/project/yfl/database/ckpt/yfl/stable_diffusion_v2-1-base &
# done

# wait
