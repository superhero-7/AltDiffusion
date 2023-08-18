# tgt_dir=metric_samples/reflow/val10k_2reflow_v2_3epoch
# gpu=0
# echo computing metrics for $tgt_dir

# # IS
# fidelity --gpu $gpu --isc --input1 $tgt_dir/IS/images

# # FID
# gt_dir=data/coco2014_reflow/val10k/content/original_images
# fidelity --gpu $gpu --fid --input1 $tgt_dir/IS/images --input2 $gt_dir

# # CLIPSim
# python CLIPSim.py \
#     --images_dir $tgt_dir/CLIPSim/images \
#     --captions_json $tgt_dir/CLIPSim/captions.json \
#     --gpu $gpu \
#     --bs 12

gt_dir=/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/gt_imgs
sample_dir=$1
clip_dir=$2
gpu=$3

# python run.py \
#     --fid \
#     --isc \
#     --sample_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/generate_imgs/laion5plus_256_60k \
#     --gt_dir $gt_dir \
#     --gpu 0 \
#     --resolution 512 \
#     --clip_sim \
#     --clip_dir /share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/laion5plus_256_60k.json;
python run.py \
     --fid \
     --isc \
    --sample_dir $sample_dir \
    --gt_dir $gt_dir \
    --gpu $gpu \
    --resolution 512 \
    --clip_sim \
    --clip_dir $clip_dir;
# python run.py \
#     --fid \
#     --isc \
#     --sample_dir /share/project/yfl/codebase/stable_diffusion_2.0/stablediffusion/nb/validation/generate_imgs/v2lg-all_ja_XX \
#     --gt_dir $gt_dir \
#     --gpu 7 \
#     --resolution 512 \
#     --clip_sim \
#     --clip_dir /share/project/yfl/codebase/stable_diffusion_2.0/stablediffusion/nb/validation/v2lg-all_ja_xx.json;
# python run.py \
#     --fid \
#     --isc \
#     --sample_dir /share/project/yfl/codebase/stable_diffusion_2.0/stablediffusion/nb/validation/generate_imgs/v2lg-all_zh_CN \
#     --gt_dir $gt_dir \
#     --gpu 7 \
#     --resolution 512 \
#     --clip_sim \
#     --clip_dir /share/project/yfl/codebase/stable_diffusion_2.0/stablediffusion/nb/validation/v2lg-all_zh_CN.json;
# python run.py \
#     --fid \
#     --isc \
#     --sample_dir /share/project/yfl/codebase/stable_diffusion_2.0/stablediffusion/nb/validation/generate_imgs/v2lg-kv \
#     --gt_dir $gt_dir \
#     --gpu 7 \
#     --resolution 512 \
#     --clip_sim \
#     --clip_dir /share/project/yfl/codebase/stable_diffusion_2.0/stablediffusion/nb/validation/v2lg-kv.json;
# python run.py \
#     --fid \
#     --isc \
#     --sample_dir /share/project/yfl/codebase/stable_diffusion_2.0/stablediffusion/nb/validation/generate_imgs/v2lg-kv_ar_AR \
#     --gt_dir $gt_dir \
#     --gpu 7 \
#     --resolution 512 \
#     --clip_sim \
#     --clip_dir /share/project/yfl/codebase/stable_diffusion_2.0/stablediffusion/nb/validation/v2lg-kv_ar_AR.json;
# python run.py \
#     --fid \
#     --isc \
#     --sample_dir /share/project/yfl/codebase/stable_diffusion_2.0/stablediffusion/nb/validation/generate_imgs/v2lg-kv_ja_XX \
#     --gt_dir $gt_dir \
#     --gpu 7 \
#     --resolution 512 \
#     --clip_sim \
#     --clip_dir /share/project/yfl/codebase/stable_diffusion_2.0/stablediffusion/nb/validation/v2lg-kv_ja_XX.json;
# python run.py \
#     --fid \
#     --isc \
#     --sample_dir /share/project/yfl/codebase/stable_diffusion_2.0/stablediffusion/nb/validation/generate_imgs/v2lg-kv_zh_CN \
#     --gt_dir $gt_dir \
#     --gpu 7 \
#     --resolution 512 \
#     --clip_sim \
#     --clip_dir /share/project/yfl/codebase/stable_diffusion_2.0/stablediffusion/nb/validation/v2lg-kv_zh_CN.json;
