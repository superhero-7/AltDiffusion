import os
import json
import argparse
from tqdm import tqdm
from pytorch_lightning import seed_everything
from diffusers import StableDiffusionPipeline

def append_data(prompts, img_names, datum, lg):
    if lg != "en":
        # 过滤一下超长的问题
        if len(datum['target_caption']) > 75:
            prompts.append(datum['target_caption'][:75])
        else:
            prompts.append(datum['target_caption'])
    else:
        prompts.append(datum['source_caption'])
    # img_names.append(datum['image'].split('/')[-1])
    img_names.append(str(datum['uni_id'])+'.jpg')
    return prompts, img_names


def main(data, scale, seed, batch_size, lg, save_path):
    seed_everything(seed)
    pipe = StableDiffusionPipeline.from_pretrained("IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1").to('cuda')
    pipe.safety_checker = lambda images, clip_input: (images, False)

    img_names = []
    prompts = []
    batch_count = 0
    for datum in tqdm(data, desc="Generating:"):
        
        batch_count += 1
        
        if batch_count < batch_size:
            append_data(prompts=prompts, img_names=img_names, datum=datum, lg=lg)
            continue
        append_data(prompts=prompts, img_names=img_names, datum=datum, lg=lg)
        
        try:
            images = pipe(prompts, guidance_scale=scale).images
        except:
            img_names = []
            prompts = []
            batch_count = 0
            continue
        for image, img_name in zip(images, img_names):
            img_save_path = save_path + '/' + img_name
            image.save(img_save_path)
            
        img_names = []
        prompts = []
        batch_count = 0



if __name__ == '__main__':
    
    save_dir = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate_imgs/'
    seed = 544421
    batch_size = 10
    scale = 9.0
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--lg', default="en", type=str)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--use_multi', action='store_true')
    parser.add_argument('--num_multi', default=8, type=int)
    
    arg = parser.parse_args()
    
    # 创建图片存储的文件夹
    if arg.lg != 'en':
        save_path = save_dir + arg.exp_name + '_' + arg.lg
    else:
        save_path = save_dir + arg.exp_name
    if not os.path.exists(save_path) and arg.rank==0:
        # 这边要判断一下是不是第一个张卡，不然会出现问题
        os.makedirs(save_path)
    
    # 读取GT的数据
    if arg.lg != "en":
        print("Loading {} data...".format(arg.lg))
        with open('/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/json_files/gt'+ '_' + arg.lg +'.json', 'r') as fn:
            data_total = json.load(fn)
    else:
        with open('/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/json_files/gt.json', 'r') as fn:
            data_total = json.load(fn)
    
    if arg.use_multi:
        length = int(len(data_total) / arg.num_multi )
        data = data_total[arg.rank*length:(arg.rank+1)*length]
    else:
        data = data_total[:]
        
    # 生成图片
    main(data=data, scale=scale, seed=seed, batch_size=batch_size, lg=arg.lg, save_path=save_path)
    
    # 存储对应的json文件
    if arg.rank==0:
        result = []
        for datum in data_total:
            
            # 这个地方计算clip score只能使用英文的数据
            caption = datum['source_caption']
            # original_dir = datum['image']
            # file_name = original_dir.split("/")[-1]
            file_name = str(datum['uni_id']) + '.jpg'
            
            if arg.lg != "en":
                result.append(
                    {
                        "caption": caption,
                        "image": '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate_imgs/' + arg.exp_name + '_' + arg.lg + '/' + file_name,
                    }
                )
            else:
                result.append(
                    {
                        "caption": caption,
                        "image": '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/coco_evaluation/generate_imgs/' + arg.exp_name + '/' + file_name,
                    }
                )

        if arg.lg != "en":
            with open('./json_files/' + arg.exp_name + '_' + arg.lg +'.json', 'w') as fn:
                json.dump(result, fn)    
        else:
            with open('./json_files/' + arg.exp_name + '.json', 'w') as fn:
                json.dump(result, fn)