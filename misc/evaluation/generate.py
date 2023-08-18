import argparse, os
import cv2
import torch
import numpy as np
import torch.distributed as dst
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder

import argparse
import json
import os
import sys
sys.path.insert(0, '/share/project/yfl/codebase/git/AltTools/Altdiffusion/src')

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def load_model_from_config(config, ckpt, verbose=False, use_ema=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    if use_ema:
        sd = pl_sd["state_dict_ema"]
    else:
        sd = pl_sd["state_dict"]
    # 模型是在这个地方初始化，初始化的
    model = instantiate_from_config(config.model)

    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def append_data(prompts, img_names, datum, lg):
    if lg != "en":
        prompts.append(datum['target_caption'])
    else:
        prompts.append(datum['source_caption'])
    img_names.append(datum['image'].split('/')[-1])
    return prompts, img_names

class OPT():
    def __init__(self) -> None:
        pass

def main(opt, data, seed, negative_prompt="", lg="", use_ema=False):

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}", use_ema=use_ema)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    elif opt.dpm:
        sampler = DPMSolverSampler(model)
    else:
        sampler = DDIMSampler(model)

    seed_everything(seed)
    batch_size = opt.n_samples
    
    img_names = []
    prompts = []
    batch_count = 0
    
    for datum in tqdm(data, desc="Gnerating:"):
        
        batch_count += 1
        
        if batch_count < batch_size:
            append_data(prompts=prompts, img_names=img_names, datum=datum, lg=lg)
            continue
        append_data(prompts=prompts, img_names=img_names, datum=datum, lg=lg)
            
        # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with HiddenPrints():
            with torch.no_grad(), \
                precision_scope(True), \
                model.ema_scope():
                    # all_samples = list()
                    # TODO: 优化一下，这个地方推理可以改成是batch的形式，现在显存才用了十几个G...
                    prompt_batch = [prompts]
                    for prompt_in in prompt_batch:
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [negative_prompt])
                        if isinstance(prompt_in, tuple):
                            prompt_in = list(prompt_in)
                        c = model.get_learned_conditioning(prompt_in)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples, _ = sampler.sample(S=opt.steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample, img_name in zip(x_samples, img_names):
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            save_path = opt.save_path + '/' + img_name
                            img.save(save_path)
        img_names = []
        prompts = []
        batch_count = 0
                

if __name__ == '__main__':
    
    opt = OPT()
    opt.prompt = "一个中国小男孩"
    opt.steps = 50
    opt.ddim_eta = 0.0
    opt.n_iter = 1
    opt.H=512
    opt.W=512
    opt.C = 4
    opt.f = 8
    opt.n_samples = 10
    opt.n_rows = 0
    opt.scale = 9.0
    opt.plms = False
    opt.dpm = False
    opt.fixed_code = False
    opt.precision = 'autocast'
    opt.seed = 54551
    save_dir = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/generate_imgs/'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--lg', default="en", type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--use_multi', action='store_true')
    parser.add_argument('--num_multi', default=8, type=int)
    parser.add_argument('--use_ema', action='store_true')
    
    arg = parser.parse_args()
    opt.config = "/share/project/yfl/codebase/git/AltTools/Altdiffusion/src/configs/v2-inference-alt.yaml"
    opt.ckpt = arg.ckpt
    
    # 创建图片存储的文件夹
    if arg.lg != 'en':
        opt.save_path = save_dir + arg.exp_name + '_' + arg.lg
    else:
        opt.save_path = save_dir + arg.exp_name
    if not os.path.exists(opt.save_path) and arg.rank==0:
        # 这边要判断一下是不是第一个张卡，不然会出现问题
        os.makedirs(opt.save_path)
    
    # 读取GT的数据
    if arg.lg != "en":
        print("Loading {} data...".format(arg.lg))
        with open('/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/json_files/gt'+ '_' + arg.lg +'.json', 'r') as fn:
            data_total = json.load(fn)
    else:
        with open('/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/json_files/gt.json', 'r') as fn:
            data_total = json.load(fn)
    
    if arg.use_multi:
        length = int(len(data_total) / arg.num_multi )
        data = data_total[arg.rank*length:(arg.rank+1)*length]
    else:
        data = data_total[:]
        
    # 生成图片
    main(opt=opt, data=data, seed=opt.seed, lg=arg.lg, use_ema=arg.use_ema)
    
    # 存储对应的json文件
    if arg.rank==0:
        result = []
        for datum in data_total:
            
            # 这个地方计算clip score只能使用英文的数据
            caption = datum['source_caption']
            original_dir = datum['image']
            file_name = original_dir.split("/")[-1]
            
            if arg.lg != "en":
                result.append(
                    {
                        "caption": caption,
                        "image": '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/generate_imgs/' + arg.exp_name + '_' + arg.lg + '/' + file_name,
                    }
                )
            else:
                result.append(
                    {
                        "caption": caption,
                        "image": '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation/generate_imgs/' + arg.exp_name + '/' + file_name,
                    }
                )

        if arg.lg != "en":
            with open('./json_files/' + arg.exp_name + '_' + arg.lg +'.json', 'w') as fn:
                json.dump(result, fn)    
        else:
            with open('./json_files/' + arg.exp_name + '.json', 'w') as fn:
                json.dump(result, fn)