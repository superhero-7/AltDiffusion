import torch_fidelity
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from glob import glob
import os
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser
import argparse
from loguru import logger
from omegaconf import OmegaConf
from geval.utils import set_seed
from geval.clip_model import get_clip
from CLIPSim import GEval

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImageDataset(Dataset):
    def __init__(self, data_root, transform=None) -> None:
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        img_paths = glob(os.path.join(data_root, '*'))
        img_paths = [p for p in img_paths if Path(
            p).suffix[1:] in IMAGE_EXTENSIONS]  # .jpg -> jpg
        self.img_paths = sorted(img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        image = Image.open(self.img_paths[i]).convert('RGB') # ! there are gray scale images in dataset
        if self.transform:
            image = self.transform(image)
        return image

def get_transforms(resolution, center_crop=False):

    transforms = []
    transforms.append(T.Resize(resolution, InterpolationMode.BILINEAR, ))
    if center_crop:
        transforms.append(T.CenterCrop(resolution))
    transforms.append(T.PILToTensor())
    transforms = T.Compose(transforms)
    return transforms


def prepare_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = ArgumentParser()
    # switches
    parser.add_argument(
        "--isc",
        action="store_true",
    )
    parser.add_argument(
        "--fid",
        action="store_true",
    )
    parser.add_argument(
        "--clip_sim",
        action="store_true",
    )
    # IS and FID
    parser.add_argument(
        "--sample_dir",
        type=str,
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
    )
    parser.add_argument(
        "--center_crop",
        type=str2bool,
        default=True,
    )
    # CLIPSim
    parser.add_argument(
        "--clip_sample_dir",
        type=str,
    )
    parser.add_argument(
        "--clip_captions_json",
        type=str,
    )
    # common
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2020,
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
    )
    parser.add_argument(
        '--clip_dir',
        type=str,
        default="",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = prepare_args()
    
    exp_name = args.sample_dir.split("/")[-1]

    logger.add(f'{args.logdir}/{exp_name}.log')

    # # ! debug
    # ###########################
    # args.gpu=0
    # args.fid=True
    # args.sample_dir='metric_samples/reflow/test/IS/images'
    # args.gt_dir='data/coco2014/val2014'
    # args.clip_sample_dir = 'metric_samples/reflow/test/CLIPSim/images'
    # args.clip_captions_json = 'metric_samples/reflow/test/CLIPSim/captions.json'
    # ###########################

    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu == - \
        1 else str(args.gpu)
    device = 'cpu' if args.gpu == -1 else 'cuda'
    logger.info(f'\n{OmegaConf.to_yaml(vars(args))}')

    # IS and FID
    if args.isc or args.fid:
        sample_ds = ImageDataset(
            args.sample_dir,
            transform=get_transforms(args.resolution)
        )
    else:
        sample_ds = None

    if args.fid:
        gt_ds = ImageDataset(
            args.gt_dir,
            transform=get_transforms(args.resolution, args.center_crop)
        )
        # # ! debug
        # for i, data in tqdm(enumerate(gt_ds)):
        #     if data.shape != torch.Size([3,256,256]):
        #         print(i, gt_ds.img_paths[i])
        #         print(data.shape)
        #         break
    else:
        gt_ds = None

    if args.isc or args.fid:
        result = torch_fidelity.calculate_metrics(
            input1=sample_ds,
            input2=gt_ds,
            isc=args.isc,
            fid=args.fid,
            batch_size=args.bs,
            rng_seed=args.seed,
            cuda=(device != 'cpu'),
        )
        logger.info(f'\n{OmegaConf.to_yaml(result)}')

    # CLIPSim
    # NOTE CLIPSim 因为有 preprocess 步骤所以不需要指定 img 的 resolution
    if args.clip_sim:
        set_seed(args.seed)
        model, preprocess, tokenizer = get_clip(device=device)
        geval = GEval(model, preprocess, tokenizer, device)
        # select_output_path = str(
        #     Path(args.clip_sample_dir).parent / 'select_best_output.json')
        # geval.select_best(
        #     images_dir=args.clip_sample_dir,
        #     captions_json=args.clip_captions_json,
        #     out_path=select_output_path,
        #     bs=args.bs,
        # )
        
        logger.info(f'overall CLIPSim {geval.CLIPSim(args.clip_dir)}')
