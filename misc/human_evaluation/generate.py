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
from glob import glob

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

def append_data(prompts, datum):
    
    prompts.append(datum)
    
    return prompts

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
    
    
    for out_idx, prompt in enumerate(tqdm(data, desc="Gnerating:")):
        
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
                prompts = [prompt]*opt.n_samples

                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [negative_prompt])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)
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

                save_path = opt.save_path + '/' + str(out_idx)
                os.makedirs(save_path, exist_ok=True)
                for inner_idx, x_sample in enumerate(x_samples):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img_save_path = save_path + '/' + str(inner_idx) + '.jpg'
                    img.save(img_save_path)
                prompt_save_path = save_path + '/prompt.txt' 
                with open(prompt_save_path, 'w') as fn:
                    fn.write(prompt)
                            
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
    opt.n_samples = 8
    opt.n_rows = 0
    opt.scale = 9.0
    opt.plms = False
    opt.dpm = False
    opt.fixed_code = False
    opt.precision = 'autocast'
    opt.seed = 54551
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--exp_name', type=str)
    # parser.add_argument('--lg', default="en", type=str)
    # parser.add_argument('--ckpt', type=str)
    # parser.add_argument('--use_ema', action='store_true')
    # parser.add_argument('--input_dir', type=str, default=None)
    # parser.add_argument('--save_dir', type=str, default=None)
    
    # arg = parser.parse_args()
    model_name = 'altdiffusion'
    opt.config = "/share/project/yfl/codebase/git/AltTools/Altdiffusion/src/configs/v2-inference-alt.yaml"
    opt.ckpt = '/share/project/yfl/database/ckpt/yfl/altdiffusion-m18-final-v2/step=000025000.ckpt'
    save_path = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/generated_results'
    input_dir = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/prompts/*'
    
    file_names = glob(input_dir)
    
    for file_name in file_names:
        lg = file_name.rsplit('_', 1)[1].split('.')[0]
    
        opt.save_path = save_path+ '/' + lg
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        
        data = []
        with open(file_name, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')  #去掉列表中每一个元素的换行符
                data.append(line)
            
        # 生成图片
        main(opt=opt, data=data, seed=opt.seed, lg=lg, use_ema=False)