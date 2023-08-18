CUDA_VISIBLE_DEVICES=0 python translate.py --path /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/json_files/gt.json --lg pt_XX &
CUDA_VISIBLE_DEVICES=1 python translate.py --path /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/json_files/gt.json --lg it_IT &
CUDA_VISIBLE_DEVICES=2 python translate.py --path /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/json_files/gt.json --lg es_XX &
wait
CUDA_VISIBLE_DEVICES=0 python translate.py --path /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/json_files/gt.json --lg de_DE &
CUDA_VISIBLE_DEVICES=1 python translate.py --path /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/json_files/gt.json --lg ru_RU 
