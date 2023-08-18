# %%
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# model_id = "stabilityai/stable-diffusion-2-1"
model_id = '/share/project/yfl/database/ckpt/yfl/stable_diffusion_v2-1-base'

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# %%
# prompt = "an oil painting of a kitten, trending on artstation, by salvador dali "
prompt = "cow"
image = pipe(prompt, height=512, width=512).images[0]



image

# %%

from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from japanese_stable_diffusion import JapaneseStableDiffusionPipeline

model_id = '/share/project/yfl/database/hub/models--rinna--japanese-stable-diffusion/snapshots/07655518e5518c6ad4340168d1d0c98958e96ae0'
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipe = JapaneseStableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler).to("cuda")
pipe.safety_checker = lambda images, clip_input: (images, False)
# %%
model_id2 = '/share/project/yfl/database/hub/models--IDEA-CCNL--Taiyi-Stable-Diffusion-1B-Chinese-v0.1/snapshots/01de965a69b5a591f0cdc35f42624b4a6ff3146e'
pipe2 = StableDiffusionPipeline.from_pretrained(model_id2).to('cuda')
# %%
import json
import pandas as pd
from glob import glob


skip_lg = ['uk', 'pl', 'ko', 'pt', 'vi']

file_names = glob('/share/project/wxy/WIT/18data/*_ocr_clip.csv')

save_path = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files'


for file_name in file_names:
    lg = file_name.split('/')[-1].split('_')[0]
    if lg not in skip_lg:
        continue
    df = pd.read_csv(file_name)
    df = df[:3400]
    
    data = []
    
    for index, row in df.iterrows():
        caption = row['caption_reference_description']
        image = row['image_path']
        data.append(
            {
                "image": image,
                "target_caption": caption,
            }
        )
    
    file_save_path = save_path + '/wit_{}'.format(lg) + '.json'
    with open(file_save_path, 'w') as fn:
        json.dump(data, fn)
# %%

json_files = glob('/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/*')

for json_file in json_files:
    d = json.load(open(json_file))
    print(len(d))

# %%
import json
# languages=["ar", "de", "es", "fr", "hi", "it", "ja", "nl", "ru", "th", "tr", "zh", "uk", "pl", "ko", "pt", "vi", "en"]
languages=["ja"]
# languages=["en"]

crossmodal_root = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/json_files/'
wit_root = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/'
joint_root = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/joint_json_files/'

for lg in languages:
    crossmodal_path = crossmodal_root+'crossmodal_'+lg+'.json'
    wit_path = wit_root+'wit_'+lg+'.json'
    
    crossmodal_data = json.load(open(crossmodal_path ,'r'))
    wit_data = json.load(open(wit_path, 'r'))
    
    joint_data = crossmodal_data + wit_data
    
    save_root = joint_root+ lg + '.json'
    with open(save_root, 'w') as fn:
        json.dump(joint_data, fn)
# %%
import os
from glob import glob
import shutil
import json


wit_image_root = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_gt_images/'
wit_root = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/*'

file_names = glob(wit_root)



for file_name in file_names:
    lg = file_name.split('_')[-1].split('.')[0]
    data = json.load(open(file_name, 'r'))
    image_save_root = wit_image_root+lg
    os.makedirs(image_save_root, exist_ok=True)
    for datum in data:
        original_image_path = datum['image']
        image_name = original_image_path.split('/')[-1]
        target_dir = image_save_root + '/' + image_name
        shutil.copy(original_image_path, target_dir)
# %%
import json

lgs = ['ja']
wit_path = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files'


for lg in lgs:
    file_path = wit_path+'/wit_'+lg+'.json'
    data = json.load(open(file_path, 'r')) 
    img_save_dir = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/ja_sd_generated_imgs/wit_ja'
    for datum in data:
        image_name = datum['sd_generated_image'].split('/')[-1]
        # datum['generated_image']= img_save_dir + '/' + image_name
        datum['japanese_sd_generated_image']= img_save_dir + '/' + image_name
        # datum['caption'] = datum['source_caption']
    with open(file_path,'w') as fn:
        json.dump(data, fn)


# %%
import json

lgs = ['zh']
root = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/joint_json_files/'

for lg in lgs:
    file_path = root+lg+'.json'
    data = json.load(open(file_path, 'r')) 
    # img_save_dir = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/ja_sd_generated_imgs/wit_ja'
    for datum in data:
        bilingual_path = datum['taiyi_generated_image']
        zh_path = bilingual_path.replace('taiyi_bilingual_generated_imgs', 'taiyi_generated_imgs')
        datum['taiyi_zh_generated_image']= zh_path
        # datum['caption'] = datum['source_caption']
    with open(file_path,'w') as fn:
        json.dump(data, fn)

# %%
