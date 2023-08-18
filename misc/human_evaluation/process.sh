# languages=("th")

# root_folder='/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/generated_results'
root_folder='/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/th'
target_folder='/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/generated_results_backup/th'
# for lg in "${languages[@]}";do
i=0
for file in "$root_folder"/*; do
    echo $i
    cp -r "$file/altdiffusion" "$target_folder/$i"
    ((i++))
    # mv "/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/generated_results/$lg/altdiffusion/altdiffusion/" "$root_folder/$lg"
done
# done