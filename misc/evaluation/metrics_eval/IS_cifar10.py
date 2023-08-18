import torch
import torch.utils.data
from geval.inception_score import inception_score 
import torchvision.datasets as dset
import torchvision.transforms as transforms

cifar10_dir = '/home/xingzhaohu/sharefs/datasets/cifar10'

if __name__ == '__main__':

    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    cifar = dset.CIFAR10(root=cifar10_dir, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(cifar), device="cuda:1", batch_size=32, resize=True, splits=10))