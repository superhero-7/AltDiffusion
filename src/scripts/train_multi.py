import os
import sys
import torch
import datetime
import argparse

sys.path.insert(0, os.getcwd())

from pytorch_lightning.trainer import Trainer
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from pytorch_lightning.loggers import WandbLogger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 主体函数部分
if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--MASTER_PORT",
        type=str,
        default="20008",
        help="DDP MASTER PORT"
    )
    parser.add_argument(
        "--MASTER_ADDR",
        type=str,
        required = True,
        help="DDP MASTER NODE port",
    )
    parser.add_argument(
        "--WORLD_SIZE",
        type=str,
        required = True,
        help="nums of nodes in cluster",
    )
    parser.add_argument(
        "--NODE_RANK",
        type=str,
        required = True,
        help="current node rank",
    )

    opt = parser.parse_args()


    os.environ["MASTER_PORT"] = opt.MASTER_PORT
    os.environ["MASTER_ADDR"] = opt.MASTER_ADDR
    os.environ["WORLD_SIZE"] = opt.WORLD_SIZE
    os.environ["NODE_RANK"] = opt.NODE_RANK


    base = "/share/project/yfl/codebase/git/AltTools/Altdiffusion/src/configs/train_multi.yaml"
    config = OmegaConf.load(base) 
    
    my_train_config = config["train"]
    
    # 设置log和ckpt保存路径
    logdir = my_train_config["logdir"]
    
    # 取出对应的字典
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    # 初始化model
    model = instantiate_from_config(config.model)
    model_dict = model.state_dict()

    # 对预训练权重进行选择性加载
    pretrained_dict = torch.load(my_train_config["pretrain_ckpt_dir"], map_location=torch.device('cpu'))["state_dict"]
    pretrained_dict = {key: value for key, value in pretrained_dict.items() if (key in model_dict and model_dict[key].shape == value.shape )}
    
    # 更新权重之后进行load
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    del pretrained_dict

    # 初始化data,注意直接在配置文件中删除掉不需要的参数
    data = instantiate_from_config(config.data)
    
    model.data = data

    # 学习率与batch_size
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    model.learning_rate = base_lr

    # 配置callback
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # 是否要恢复训练
    resume = False

    # checkpoint的保存设置
    # default_modelckpt_cfg = {
    #     'metrics_over_trainsteps_checkpoint':{
    #     "target": "pytorch_lightning.callbacks.ModelCheckpoint",
    #     "params": {
    #         "dirpath": ckptdir,
    #         "filename": "{epoch:09}",
    #         "every_n_epochs": 1,
    #         "save_top_k": -1,
    #     }
    #     },
    # }

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
        # "image_logger": {
        #     "target": "main.ImageLogger",
        #     "params": {
                
        #         "batch_frequency": 3000,
        #         "logdir":logdir,
        #         "max_images": 4,
        #         "clamp": True
        #     }
        # },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
               
                "logging_interval": "step"
                # "log_momentum": True
            }
        },
    }

    # trainer and callbacks config
    trainer_kwargs = dict()
    # 创建trainer
    callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    # callbacks_cfg = OmegaConf.merge(default_modelckpt_cfg, callbacks_cfg)
    # 给trainer设置chekpoint和log的callback
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    
    
    # 设置训练的epoch
    trainer_kwargs["max_epochs"] = my_train_config["max_epochs"]
    
    # 设置gpu使用和是否多机训练
    trainer_kwargs["accelerator"] = my_train_config["ddp"]["accelerator"]
    trainer_kwargs["devices"] = my_train_config["ddp"]["devices"]
    trainer_kwargs["strategy"] = my_train_config["ddp"]["strategy"]
    trainer_kwargs["num_nodes"] = my_train_config["ddp"]["num_nodes"]
    trainer_kwargs["accumulate_grad_batches"] = my_train_config["accumulate_grad_batches"]
    # trainer_kwargs["enable_checkpointing"] = False
    
    # trainer_kwargs["logdir"] = logdir

    trainer = Trainer.from_argparse_args(trainer_config,**trainer_kwargs)

    # 使用wandb
    # wandb_logger = WandbLogger(project="stable_diffusion",name='sd2.0' ,save_dir=logdir)
    # trainer.logger = wandb_logger
    
    # 开始训练
    trainer.fit(model, train_dataloaders=data.data['train'].dataloader)