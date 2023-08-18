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

class OPT():
    def __init__(self) -> None:
        pass

def main(opt, data, seeds, negative_prompt="", lg="", use_ema=False, exp_name=""):

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

    batch_size = opt.n_samples
    
    img_count = len(os.listdir(opt.save_path))

    for idx, prompt in enumerate(tqdm(data, desc="Gnerating:")):
        
        prompt = prompt.strip('\n')
        
        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        
        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        
        for i in range(8):
            seed = seeds[i]
            seed_everything(seed)
            
            with HiddenPrints():
                with torch.no_grad(), \
                    precision_scope(True), \
                    model.ema_scope():
                        # all_samples = list()
                        # TODO: 优化一下，这个地方推理可以改成是batch的形式，现在显存才用了十几个G...
                        prompt_batch = [batch_size * [prompt]]
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

                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                # save_path = opt.save_path + 'baaicn-' + str(idx) + '/'
                                save_path = opt.save_path + f'img-{img_count:04}' + '/'
                                os.makedirs(save_path, exist_ok=True)
                                # img.save(save_path+exp_name+'.jpg')
                                img.save(save_path+str(i)+'.jpg')
        img_count += 1
        with open(save_path+'prompts.txt','w',encoding='utf8') as f:
            f.write(prompt)
                    

if __name__ == '__main__':
    
    opt = OPT()
    opt.prompt = "一个中国小男孩"
    opt.steps = 50
    opt.ddim_eta = 0.0
    opt.n_iter = 1
    opt.H=256
    opt.W=256
    opt.C = 4
    opt.f = 8
    opt.n_samples = 1
    opt.n_rows = 0
    opt.scale = 9.0
    opt.plms = False
    opt.dpm = False
    opt.fixed_code = False
    opt.precision = 'autocast'
    opt.seeds = [377558312, 654444536, 563203535, 21588556, 1008617945, 1122612923, 1023025838, 1126986233]
    save_dir = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/visualization/generated_imgs/'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--prompt_dir', type=str)
    arg = parser.parse_args()
    opt.config = "/share/project/yfl/codebase/git/AltTools/Altdiffusion/src/configs/v2-inference-alt.yaml"
    opt.ckpt = arg.ckpt
    
    prompts_dir = arg.prompt_dir
    
    # 创建图片存储的文件夹
    opt.save_path = save_dir + arg.exp_name + '/'
    os.makedirs(opt.save_path, exist_ok=True)
    
    # 读取数据
    with open(prompts_dir, 'r') as fn:
        data = fn.readlines()
    
    negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

    # 生成图片
    main(opt=opt, negative_prompt=negative_prompt, data=data, seeds=opt.seeds, exp_name=arg.exp_name)