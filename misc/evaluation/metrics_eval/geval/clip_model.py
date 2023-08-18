import torch
from PIL import Image
from loguru import logger
import geval.open_clip as open_clip 

# 模型的 url 在 open_clip.pretrained 文件中
def get_clip(model_name="ViT-B-32-quickgelu", 
                pretrained="laion400m_e32", 
                cache_dir="./cached_models/", 
                device='cpu'):
    
    logger.info(f"loading model from <<{cache_dir}>>")
    logger.info(f"model name <<{model_name}>> ; pretrained dataset <<{pretrained}>>")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, 
                                                                pretrained=pretrained, 
                                                                cache_dir=cache_dir,
                                                                device=device)
                                                                
    tokenizer = open_clip.get_tokenizer(model_name)

    logger.info(f"load model done....")
    return model, preprocess, tokenizer

