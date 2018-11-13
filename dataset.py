import glob
import os
import random

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImgPairDataset(Dataset):
    def __init__(self, dataroot, transforms, modeStr):
        self.dataroot = dataroot
        self.transforms = transforms
        self.datasetPathsA = sorted(glob.glob(os.path.join(dataroot, modeStr+'A') + '/*.*'))
        self.datasetPathsB = sorted(glob.glob(os.path.join(dataroot, modeStr+'B') + '/*.*'))
        self.lenA = len(self.datasetPathsA)
        self.lenB = len(self.datasetPathsB)

    def __len__(self):
        return max(len(self.datasetPathsA), len(self.datasetPathsB))

    def  __getitem__(self, idx):
        imgA = self.transforms(Image.open(self.datasetPathsA[idx % self.lenA]))
        imgB = self.transforms(Image.open(self.datasetPathsB[idx % self.lenB]))

        return {'A': imgA, 'B': imgB}