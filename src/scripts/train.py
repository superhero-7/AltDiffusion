import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import datetime
# import wandb

from pytorch_lightning.trainer import Trainer
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from pytorch_lightning.loggers import WandbLogger

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"]= '1'

# 主体函数部分
if __name__ == "__main__":

    # 读取对应的配置文件，得到相应的字典
    base = "configs/train.yaml"
    config = OmegaConf.load(base) 
    
    # 取出对应的字典
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    # 初始化model
    model = instantiate_from_config(config.model)
    model_dict = model.state_dict()

    # 对预训练权重进行选择性加载
    pretrained_dict = torch.load("/share/project/yfl/codebase/stable_diffusion_2.0/ckpt/v2-1_512-ema-pruned.ckpt")["state_dict"]
    pretrained_dict = {key: value for key, value in pretrained_dict.items() if (key in model_dict and model_dict[key].shape == value.shape )}

    # 更新权重之后进行load
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    
    # 初始化data,注意直接在配置文件中删除掉不需要的参数
    data = instantiate_from_config(config.data)
    
    model.data = data
    
    # 学习率与batch_size
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    model.learning_rate = base_lr

    # 设置log和ckpt保存路径
    logdir = config.logdir
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    # ligthning 的config
    lightning_config = config.pop("lightning", OmegaConf.create())
    
    # 是否要恢复训练
    resume = False 

    # 给trainer设置chekpoint和log的callback
    # checkpoint的保存设置
    default_modelckpt_cfg = {
        'metrics_over_trainsteps_checkpoint':{
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{step:09}",
            "every_n_train_steps": 5,
            "save_top_k": -1,
        }
        },
    }

    # log 的回调函数定义
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "callbacks.CallbackLogger.SetupCallback",
            "params": {
                "resume": resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config":lightning_config
            }
        },
        "image_logger": {
            "target": "callbacks.CallbackLogger.ImageLogger",
            "params": {
                "batch_frequency": 30000,
                "max_images": 1,
                "logdir": logdir,
                "clamp": True
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            }
        },
    }
    
    # trainer and callbacks config
    trainer_kwargs = dict()
    # 创建trainer
    callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    callbacks_cfg = OmegaConf.merge(default_modelckpt_cfg, callbacks_cfg)

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    
    # 设置训练的epoch
    trainer_kwargs["max_epochs"] = 2
    
    # 设置gpu使用和是否多机训练
    trainer_kwargs["accelerator"] = 'gpu'
    trainer_kwargs["devices"] = 2
    trainer_kwargs["strategy"] = "ddp"
    
    # 使用wandb
    # wandb_logger = WandbLogger(project="m18")
    # trainer_kwargs["logger"] = wandb_logger

    trainer = Trainer.from_argparse_args(trainer_config,**trainer_kwargs)
    trainer.logdir = logdir

    # 开始训练
    trainer.fit(model, train_dataloaders=data.data['train'].dataloader)
    
    # from torch.distributed import init_process_group, get_rank, get_world_size
    
    # os.environ["ENV_TYPE"] = "deepspeed+mpu"
    # model_parallel_size = 1
    # world_size = 2

    # os.environ["MODEL_PARALLEL_SIZE"] = str(model_parallel_size)
    # os.environ["WORLD_SIZE"] = str(world_size)

    # def set_random_seed(seed):
    #     """Set random seed for reproducability."""
    #     if seed is not None and seed > 0:
    #         random.seed(seed)
    #         np.random.seed(seed)
    #         torch.manual_seed(seed)
    #         mpu.model_parallel_cuda_manual_seed(seed)
    # import argparse
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--local_rank',
    #                     type=int,
    #                     default=0,
    #                     help="local_rank")

    # ds_args = parser.parse_args()
    # local_rank = ds_args.local_rank

    # master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    # master_port = os.environ.get('MASTER_PORT', '17501')

    # device = torch.device("cuda", local_rank)

    # def initialize_distributed():
    #     """Initialize torch.distributed."""
    #     torch.backends.cudnn.enabled = False
    #     # Manually set the device ids.
    #     torch.cuda.set_device(device)
    #     # Call the init process
    #     init_method = 'tcp://'

    #     init_method += master_addr + ':' + master_port
    #     torch.distributed.init_process_group(
    #         backend='nccl',  # gloo
    #         world_size=world_size,
    #         rank=local_rank,
    #         init_method=init_method)
    #     # mpu.initialize_model_parallel(model_parallel_size)

    # initialize_distributed()
    
    # for idx, it in enumerate(data.data["train"].dataloader):
    #     import torch.distributed as dst
    #     rank = dst.get_rank()
    #     print("rank {} get {} Text is {}".format(rank, idx, it[1][0]))
    #     continue
        
    
    
    