import torch
from PIL import Image
from geval.clip_model import get_clip
import json 
import glob 
import os 
from loguru import logger
from geval.utils import set_seed
from argparse import Namespace, ArgumentParser 
from pathlib import Path
from tqdm.auto import tqdm

class GEval:
    def __init__(self, model, preprocess, tokenizer, device='cpu') -> None:
        self.model = model 
        self.preprocess = preprocess 
        self.tokenizer = tokenizer
        self.device=device
    
    def compute_sim(self, image_path, caption):
        if isinstance(image_path, list):
            image = [self.preprocess(Image.open(p)) for p in image_path]
            image=torch.stack(image, dim=0)
        else:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0)
        text = self.tokenizer([caption])

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image.to(self.device))
            text_features = self.model.encode_text(text.to(self.device))
            
            text_probs = torch.nn.functional.cosine_similarity(image_features, text_features, dim=1)

        return text_probs

    def select_best(self, images_dir, captions_json, out_path="select_outputs.json", bs=1): 
        out = {}
        
        def devide_by_bs(seq:list, bs):
            groups = []
            now_idx=0
            total=len(seq)
            while now_idx<total:
                s,e=now_idx,now_idx+bs
                if bs>total:
                    bs=total
                groups.append(seq[s:e])
                now_idx+=bs
            return groups
        
        def argmax(seq:list):
            return seq.index(max(seq))

        with open(captions_json) as f :
            data = json.loads(f.read())
        
        for k, v in tqdm(data.items()):
            caption = v 
            cur_images_dir = os.path.join(images_dir, k)
            all_images = sorted(glob.glob(f"{cur_images_dir}/*.png"))
            if len(all_images)==0:
                raise(ValueError(f'find 0 png images in {cur_images_dir}'))
            groups=devide_by_bs(all_images, bs)
            all_prob = []
            for image_paths in groups:
                probs = self.compute_sim(image_paths, caption)
                all_prob.extend( probs.tolist() )
            # for image in all_images:
            #     prob = self.compute_sim(image, caption)
            #     all_prob.append(prob)
            
            # logger.info(f"caption is {v}, sim_list is {all_prob}")
            out[k] = {
                "caption": v,
                "best_image": all_images[argmax(all_prob)]
            }
        json.dump(out, open(out_path, 'w'))
        logger.info(f"select best images done, output path is {out_path}")

    def CLIPSim(self, data_json):
        data = json.load(open(data_json))
        # with open(data_json) as f:
        #     data = json.loads(f.read())
        
        all_sim = []
        for datum in tqdm(data):
            caption = datum["caption"]
            image = datum["image"]
            try:
                sim = self.compute_sim(image, caption)
            except Exception as e:
                print(e)
                continue
            all_sim.append(sim.item())

        return sum(all_sim) / len(all_sim)


def prepare_args():
    args = Namespace()
    args.images_dir = 'data_example/images'
    args.captions_json = 'data_example/data.json'
    args.seed = 23
    args.gpu = -1
    
    parser = ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str
    )
    parser.add_argument(
        "--captions_json",
        type=str
    )
    parser.add_argument(
        "--gpu",
        type=int, 
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=1,
    )
    parser.parse_args(namespace=args)
    
    return args

if __name__ == "__main__":
    args=prepare_args()
    root_dir = Path(args.images_dir).parent
    logger.add(str(root_dir / 'run.log'))
    
    set_seed(args.seed)
    logger.info(f'use random seed {args.seed}')

    device = 'cpu' if args.gpu==-1 else f'cuda:{args.gpu}'
    logger.info(f'process running in {device}')
    model, preprocess, tokenizer = get_clip(device=device)

    geval = GEval(model, preprocess, tokenizer, device)

    select_output_path = str(Path(args.images_dir).parent / 'select_best_output.json')
    geval.select_best(
        images_dir=args.images_dir,
        captions_json=args.captions_json,
        out_path=select_output_path,
        bs=args.bs,
        )

    logger.info(f'overall CLIPSim {geval.CLIPSim(select_output_path)}')
