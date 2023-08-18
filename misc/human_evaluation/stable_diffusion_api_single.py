import os
import torch
import json
import argparse
from tqdm import tqdm
from glob import glob
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler


class Stable_Diffusion_Inference_API():
    
    def __init__(self, config) -> None:
        
        self.config = config
        self.pipe = StableDiffusionPipeline.from_pretrained(self.config.model_base)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to("cuda")

    def inference(self, prompts):
        
        images = self.pipe(prompts, 
            height=self.config.height,
            width=self.config.width, 
            num_inference_steps=self.config.inference_steps,
            guidance_scale=self.config.guidance_scale)
            
        
        return images[0]

def append_data(prompts, img_names, datum, lg):
    # if lg != "en":
    #     prompts.append(datum['target_caption'])
    # else:
    prompts.append(datum['source_caption'])
    img_names.append(datum['image'].split('/')[-1])
    return prompts, img_names

class Args():
    def __init__(self) -> None:
        pass

if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser()
    # # parser.add_argument('--exp_name', type=str)
    # parser.add_argument('--lg', default=None, type=str)
    # parser.add_argument('--model_base', default="/share/project/yfl/database/ckpt/yfl/stable_diffusion_v2-1", type=str)
    # parser.add_argument('--height', default=512, type=int)
    # parser.add_argument('--width', default=512, type=int)
    # parser.add_argument('--inference_steps', default=50, type=int)
    # parser.add_argument('--guidance_scale', default=9.0)
    # parser.add_argument('--input_dir', default=None, type=str)
    # parser.add_argument('--save_dir', default=None, type=str)
    # parser.add_argument('--batch_size', default=10, type=int)
    # parser.add_argument('--lg1', default=None, type=str)
    # parser.add_argument('--lg2', default=None, type=str)
    # config = parser.parse_args()
    
    lg_list = ["vi"]
    
    args = Args()
    args.model_base = "/share/project/yfl/database/ckpt/yfl/stable_diffusion_v2-1-base"
    save_path = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/generated_results_backup'
    args.input_dir = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/translated_prompts/*'
    args.batch_size = 8
    args.height = 512
    args.width = 512
    args.inference_steps = 50
    args.guidance_scale = 9.0
    args.zh_input_dir = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/translated_prompts_zh'
    
    
    model = Stable_Diffusion_Inference_API(args)
    
    file_names = glob(args.input_dir)
    
    for file_name in file_names:
        lg = file_name.rsplit('_', 1)[1].split('.')[0]
        # 如果不在就跳过
        if lg not in lg_list:
            continue
        print("Generating {}".format(lg))
        args.save_path = save_path+ '/' + lg
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        
        data = []
        with open(file_name, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')  #去掉列表中每一个元素的换行符
                data.append(line)
            
        real_file_name = file_name.split('/')[-1]
        zh_file_path = args.zh_input_dir + '/' + real_file_name
        
        zh_data = []
        with open(zh_file_path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')  #去掉列表中每一个元素的换行符
                zh_data.append(line)
        
            
        for out_index, prompt in enumerate(tqdm(data, desc="Generating:")):
            
            zh_prompt = zh_data[out_index]
            
            prompts = [prompt]*args.batch_size
            images = model.inference(prompts)

            final_save_path = args.save_path + '/' + str(out_index) + '/sd2.1' 
            os.makedirs(final_save_path, exist_ok=True)
            for inner_index, image in enumerate(images):
                img_save_path = final_save_path + '/' + str(inner_index) + '.jpg'
                image.save(img_save_path)
            prompt_save_path = final_save_path + '/prompt.txt' 
            with open(prompt_save_path, 'w') as fn:
                fn.write(prompt)
            
            zh_prompt_save_path = args.save_path + '/' + str(out_index) + '/prompt.txt' 
            with open(zh_prompt_save_path, 'w') as fn:
                fn.write(zh_prompt)