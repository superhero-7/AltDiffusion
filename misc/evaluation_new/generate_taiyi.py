import os
import torch
import json
import argparse
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
# from japanese_stable_diffusion import JapaneseStableDiffusionPipeline

class Japanese_Stable_Diffusion_Inference_API():
    
    def __init__(self, config) -> None:
        
        self.config = config
        self.pipe = StableDiffusionPipeline.from_pretrained(self.config.model_base).to('cuda')
        # self.pipe.safety_checker = lambda images, clip_input: (images, False)
        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)
        # self.pipe.to("cuda")

    def inference(self, prompts):
        
        images = self.pipe(prompts, 
            height=self.config.height,
            width=self.config.width, 
            guidance_scale=self.config.guidance_scale)
            
        
        return images[0]

def append_data(prompts, img_names, datum, lg):
    # if lg != "en":
    #     prompts.append(datum['target_caption'])
    # else:
    prompts.append(datum['target_caption'])
    img_names.append(datum['image'].split('/')[-1])
    return prompts, img_names


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--exp_name', type=str)
    parser.add_argument('--lg', default=None, type=str)
    parser.add_argument('--model_base', default="/share/project/yfl/database/ckpt/yfl/stable_diffusion_v2-1", type=str)
    parser.add_argument('--height', default=512, type=int)
    parser.add_argument('--width', default=512, type=int)
    parser.add_argument('--inference_steps', default=50, type=int)
    parser.add_argument('--guidance_scale', default=9.0)
    parser.add_argument('--input_dir', default=None, type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--batch_size', default=10, type=int)
    args = parser.parse_args()
    
    model = Japanese_Stable_Diffusion_Inference_API(args)
    
    
    data_total = json.load(open(args.input_dir))
    
    batch_size = args.batch_size
    img_names = []
    prompts = []
    batch_count = 0
    lg = args.lg
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    for datum in tqdm(data_total, desc="Generating:"):
        
        batch_count += 1
        
        if batch_count < batch_size:
            append_data(prompts=prompts, img_names=img_names, datum=datum, lg=lg)
            continue
        append_data(prompts=prompts, img_names=img_names, datum=datum, lg=lg)
        
        
        images = model.inference(prompts)
    
        for image, img_name in zip(images, img_names):
            save_path = args.save_dir + '/' + img_name
            image.save(save_path)
        
        img_names = []
        prompts = []
        batch_count = 0
    
    
    
    for datum in data_total:
        
        # 这个地方计算clip score只能使用英文的数据
        # caption = datum['source_caption']
        original_dir = datum['image']
        file_name = original_dir.split("/")[-1]
        
        datum['taiyi_generated_image'] = args.save_dir + '/' + file_name

    with open(args.input_dir, 'w') as fn:
        json.dump(data_total, fn)