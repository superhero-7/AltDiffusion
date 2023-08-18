# languages=("zh")

# declare -A gpu_map
# gpu_map=( ["zh"]=1)

# # 启动所有后台任务
# for lg in "${languages[@]}"
# do
#     gpu=${gpu_map[$lg]}
#     CUDA_VISIBLE_DEVICES=$gpu python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/generate_taiyi.py \
#         --lg $lg \
#         --input_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/wit_${lg}.json \
#         --save_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/taiyi_generated_imgs/wit_${lg} \
#         --model_base /share/project/yfl/database/hub/models--IDEA-CCNL--Taiyi-Stable-Diffusion-1B-Chinese-v0.1/snapshots/01de965a69b5a591f0cdc35f42624b4a6ff3146e
# done

# wait

languages=("zh" "en")

declare -A gpu_map
gpu_map=( ["zh"]=0 ["en"]=1)

# 启动所有后台任务
for lg in "${languages[@]}"
do
    gpu=${gpu_map[$lg]}
    CUDA_VISIBLE_DEVICES=$gpu python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/generate_taiyi.py \
        --lg $lg \
        --input_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/wit_${lg}.json \
        --save_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/taiyi_bilingual_generated_imgs/wit_${lg} \
        --model_base /share/project/yfl/database/hub/models--IDEA-CCNL--Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/snapshots/4a433ba11ed455dd030e46bbf13f1eef578474c6 &
done

languages=("zh" "en")

declare -A gpu_map
gpu_map=( ["zh"]=2 ["en"]=3)

# 启动所有后台任务
for lg in "${languages[@]}"
do
    gpu=${gpu_map[$lg]}
    CUDA_VISIBLE_DEVICES=$gpu python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/generate_taiyi.py \
        --lg $lg \
        --input_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/json_files/crossmodal_${lg}.json \
        --save_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/taiyi_bilingual_generated_imgs/crossmodal_${lg} \
        --model_base /share/project/yfl/database/hub/models--IDEA-CCNL--Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/snapshots/4a433ba11ed455dd030e46bbf13f1eef578474c6 &
done

wait