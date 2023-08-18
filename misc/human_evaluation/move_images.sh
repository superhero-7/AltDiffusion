languages=("th")

for lg in "${languages[@]}";do
root_folder="/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/generated_results/$lg"
echo "Processing : $lg"
# 使用 find 命令遍历文件夹底下的所有子文件夹
while IFS= read -r folder; do
    # 在这里执行你想要对每个文件夹执行的操作
    # mkdir -p "$folder/altdiffusion" 
    # 移动当前子文件夹中的所有图片到 altdiffusion 子文件夹
    # mv "$folder/altdiffusion/altdiffusion"/*.txt "$folder/altdiffusion/"
    echo "Processing folder: $folder"
done < <(find "$root_folder" -type d)
done