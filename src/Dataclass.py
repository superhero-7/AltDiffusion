import os
import glob
import time
import json 
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as transforms

from PIL import Image
from data import get_data
from torch.utils.data import DataLoader, Dataset, IterableDataset

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class Train(Dataset):
    def __init__(self):

        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

        # 读取images_data 数据
        # with open("/share/project/yfl/codebase/nb/wudaomm_laion_5.5+6_new_platform.json","r") as f:
        #     self.art_data = json.load(f)
        # self.art_data = []
        # import torch.distributed as dst
        # rank = dst.get_rank()
        # file_name = "/share/project/yfl/codebase/nb/laion6plus/slices_96/slice_" + str(rank) + '.json'
        # with open(file_name, 'r') as fn:
        #     self.art_data = json.load(fn)
        
        self.art_data = []
        file_names = glob.glob("/share/project/yfl/codebase/nb/laion6plus/part_*_en.json")
        print("Loading laion 6plus data...")     
        start = time.time()
        for file_name in file_names:
            with open(file_name, 'r') as fn:
                d = json.load(fn)
            self.art_data += d
        end = time.time()
        time_sum = end-start
        print("Loaded!^_^Spend {} s... The length of training dataset is {}".format(time_sum, len(self.art_data)))
        
        self.len = len(self.art_data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        d = {}
        try:
            # caption = self.art_data[index]['caption']
            if 'target_caption' in self.art_data[index]:
                caption = self.art_data[index]['target_caption']
            else:
                caption = self.art_data[index]['source_caption']
            assert len(caption)<75
            d["caption"] = caption

            img = Image.open(self.art_data[index]['image']).convert("RGB")
            min_size = min(img.size[0],img.size[1])
            max_size = max(img.size[0],img.size[1])


            tf = transforms.Compose( [
                    transforms.RandomCrop((min_size,min_size)),
                    transforms.Resize((512,512)),
                    transforms.ToTensor(),
                    self.normalize
                ] )
            d["image"] = tf(img) 

            return d
        except Exception as e:
            # print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))
        
class Train_onegpu(Dataset):
    def __init__(self):

        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

        # 读取images_data 数据
        with open("/share/project/yfl/codebase/nb/wudaomm_laion_5.5+6_new_platform.json","r") as f:
            self.art_data = json.load(f)
        
        self.len = len(self.art_data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        d = {}
        try:
            caption = self.art_data[index]['caption']
            # if 'target_caption' in self.art_data[index]:
            #     caption = self.art_data[index]['target_caption']
            # else:
            #     caption = self.art_data[index]['source_caption']
            assert len(caption)<75
            d["caption"] = caption

            img = Image.open(self.art_data[index]['image']).convert("RGB")
            min_size = min(img.size[0],img.size[1])
            max_size = max(img.size[0],img.size[1])


            tf = transforms.Compose( [
                    transforms.RandomCrop((min_size,min_size)),
                    transforms.Resize((512,512)),
                    transforms.ToTensor(),
                    self.normalize
                ] )
            d["image"] = tf(img) 

            return d
        except Exception as e:
            # print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

class Test(Dataset):
    def __init__(self,config=None):

        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

        # 读取images_data 数据
        with open("/share/project/yfl/codebase/nb/wudaomm_laion_5.5+6_new_platform.json","r") as f:
            self.art_data = json.load(f)[-500:]

        self.len = len(self.art_data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        d = {}
        try:
            caption = self.art_data[index]['caption']
            assert len(caption)<75
            d["caption"] = caption
      
            img = Image.open(self.art_data[index]['image']).convert("RGB")
            min_size = min(img.size[0],img.size[1])
            max_size = max(img.size[0],img.size[1])


            tf = transforms.Compose( [
                    transforms.RandomCrop((min_size,min_size)),
                    transforms.Resize((512,512)),
                    transforms.ToTensor(),
                    self.normalize
                ])
            d["image"] = tf(img) 

        except Exception as e:
            return self.__getitem__(np.random.randint(0, self.__len__()-1))
            
        return d 

class DummyDataLoader():
    def __init__(self, batch_size, num_workers) -> None:
        
        dataset = Train()
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                           num_workers = num_workers,
                           shuffle=True)

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = {}
        self.data["train"] = DummyDataLoader(batch_size=batch_size, num_workers=num_workers)

    def train_dataloader(self):
        #train_dataset = WudaoTrain()
        train_dataset = Train()

        return DataLoader(train_dataset, batch_size=self.batch_size,
                           num_workers = self.num_workers,
                           shuffle=True)

    def val_dataloader(self):
        #test_dataset = WudaoTest()
        test_dataset = Test()
        return DataLoader(test_dataset,
                          batch_size=self.batch_size,
                          num_workers = self.num_workers,
                          shuffle=True)
        
class DataModuleFromConfig_onegpu(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        #train_dataset = WudaoTrain()
        train_dataset = Train_onegpu()

        return DataLoader(train_dataset, batch_size=self.batch_size,
                           num_workers = self.num_workers,
                           shuffle=True)

    def val_dataloader(self):
        #test_dataset = WudaoTest()
        test_dataset = Test()
        return DataLoader(test_dataset,
                          batch_size=self.batch_size,
                          num_workers = self.num_workers,
                          shuffle=True)

class Args():
    def __init__(self) -> None:
        pass

def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


class WebDatasetFromConfig(pl.LightningDataModule):
    def __init__(self, train_data=None, train_num_samples=600000000, val_data=None, val_num_samples=1000, 
                seed=34070, batch_size=4, workers=6, world_size=2, dataset_resample=True, start_epoch=0, 
                use_ourdecoder=False, use_256=True, use_combine=False, cfg=0) -> None:
        super().__init__()
        
        # 设置open clip data需要的参数
        arg = Args()
        arg.train_data = train_data
        arg.val_data = val_data
        arg.val_num_samples = val_num_samples
        arg.train_num_samples = train_num_samples
        arg.seed = seed
        arg.batch_size = batch_size
        arg.workers = workers
        arg.dataset_resampled = dataset_resample
        # arg.local_rank, arg.rank, arg.world_size = world_info_from_env()
        # 手动指定world_size
        arg.world_size = world_size
        arg.use_ourdecoder = use_ourdecoder
        arg.use_256 = use_256
        arg.use_combine = use_combine
        arg.cfg = cfg
        
        self.data = get_data(arg, epoch=start_epoch)
    
    def train_dataloader(self):
        return self.data['train'].dataloader
    
    