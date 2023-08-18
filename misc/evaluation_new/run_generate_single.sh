languages=("uk" "pl" "ko" "pt")

declare -A gpu_map
gpu_map=( ["uk"]=0 ["pl"]=1 ["ko"]=2 ["pt"]=3)


# 启动所有后台任务
for lg in "${languages[@]}"
do
    gpu=${gpu_map[$lg]}
    CUDA_VISIBLE_DEVICES=$gpu python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/generate.py \
        --rank 0 \
        --lg $lg \
        --input_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/wit_${lg}.json \
        --save_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/generated_imgs/wit_${lg} \
        --ckpt /share/project/yfl/database/ckpt/yfl/altdiffusion-m18-final-v2/step=000025000.ckpt &
done

wait

# languages=("fr" "hi" "it" "ja")

# declare -A gpu_map
# gpu_map=( ["fr"]=0 ["hi"]=1 ["it"]=2 ["ja"]=3)


# # 启动所有后台任务
# for lg in "${languages[@]}"
# do
#     gpu=${gpu_map[$lg]}
#     CUDA_VISIBLE_DEVICES=$gpu python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/generate.py \
#         --rank 0 \
#         --lg $lg \
#         --input_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/wit_${lg}.json \
#         --save_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/generated_imgs/wit_${lg} \
#         --ckpt /share/project/yfl/database/ckpt/yfl/altdiffusion-m18-final-v2/step=000025000.ckpt &
# done

# wait

# languages=("nl" "ru" "th" "tr")

# declare -A gpu_map
# gpu_map=( ["nl"]=0 ["ru"]=1 ["th"]=2 ["tr"]=3)


# # 启动所有后台任务
# for lg in "${languages[@]}"
# do
#     gpu=${gpu_map[$lg]}
#     CUDA_VISIBLE_DEVICES=$gpu python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/generate.py \
#         --rank 0 \
#         --lg $lg \
#         --input_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/wit_${lg}.json \
#         --save_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/generated_imgs/wit_${lg} \
#         --ckpt /share/project/yfl/database/ckpt/yfl/altdiffusion-m18-final-v2/step=000025000.ckpt &
# done

# wait

languages=("vi")

declare -A gpu_map
gpu_map=( ["vi"]=0)


# 启动所有后台任务
for lg in "${languages[@]}"
do
    gpu=${gpu_map[$lg]}
    CUDA_VISIBLE_DEVICES=$gpu python /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/generate.py \
        --rank 0 \
        --lg $lg \
        --input_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/wit_${lg}.json \
        --save_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/generated_imgs/wit_${lg} \
        --ckpt /share/project/yfl/database/ckpt/yfl/altdiffusion-m18-final-v2/step=000025000.ckpt &
done