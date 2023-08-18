languages=("uk" "pl" "ko" "pt" "vi")

declare -A lg_code_map
lg_code_map=(["uk"]="uk_UA" ["pl"]="pl_PL" ["ko"]="ko_KR" ["pt"]="pt_XX" ["vi"]="vi_VN")

# 启动所有后台任务
for lg in "${languages[@]}"
do
    lg_code=${lg_code_map[$lg]}
    CUDA_VISIBLE_DEVICES=0 python translate.py --path /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/wit_${lg}.json --lg $lg_code;
done