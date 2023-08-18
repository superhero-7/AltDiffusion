import torch
import random
import numpy as np

from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast

import sys
sys.path.insert(0, '/share/project/yfl/codebase/git/AltTools/Altdiffusion/src')
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    
    sd = pl_sd["state_dict"]
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

config_dir = "/share/project/yfl/codebase/git/AltTools/Altdiffusion/src/configs/v2-inference-alt.yaml"
# opt.ckpt = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/ckpt/xformer_laion5plus_512_kv_cfg/checkpoints/step=000015000.ckpt'
ckpt = '/share/project/yfl/database/ckpt/aethetics_all_ema_cfg/step=000025000.ckpt'
config = OmegaConf.load(f"{config_dir}")
model = load_model_from_config(config, f"{ckpt}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
sampler = DDIMSampler(model)

def generate(prompt, negative_prompt, batch_size, W, H, seed):
    seed_everything(seed)

    start_code = None
    if seed == -1:
        seed = random.randint(0, 1172684057) + 7860

    precision_scope = autocast
    with torch.no_grad(), \
        precision_scope(True), \
        model.ema_scope():
            all_samples = list()
            prompts = [batch_size * [prompt]]
            for prompts in tqdm(prompts, desc="data"):
                # uc = None
                uc = model.get_learned_conditioning(batch_size * [negative_prompt])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)
                shape = [4, H // 8, W // 8]
                samples, _ = sampler.sample(S=50,
                                                    conditioning=c,
                                                    batch_size=batch_size,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=5.0,
                                                    unconditional_conditioning=uc,
                                                    eta=0.0,
                                                    x_T=start_code)

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    all_samples.append(np.array(img))

    return all_samples
            
            
import gradio as gr

demo = gr.Interface(
    fn=generate, 
    inputs=[
        "text",
        "text",
        gr.Slider(0, 4, value=1, step=1, label="Sample size"),
        gr.Slider(512,1024,value=512, step=64, label="width"),
        gr.Slider(512,1024,value=512, step=64, label="height"),
        gr.Number(-1, label='seed', interactive=True)], 
    outputs=gr.Gallery(label="Generated Images"))

demo.queue(concurrency_count=1)

demo.launch()