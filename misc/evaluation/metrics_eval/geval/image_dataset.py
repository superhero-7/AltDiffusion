

from torch.utils.data import Dataset, DataLoader
import glob 
from PIL import Image
import numpy as np 

class ImageDataset(Dataset):
    def __init__(self, image_dir, image_type="png") -> None:
        super().__init__()

        all_images = glob.glob(f"{image_dir}/*.{image_type}")

        print(f"images length is {len(all_images)}")

        self.all_images = all_images
    
    def __getitem__(self, index):
        image = Image.open(self.all_images[index])
        image = np.array(image).astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = (image - image.mean()) / image.std()

        return image 
    
    def __len__(self):
        return len(self.all_images)