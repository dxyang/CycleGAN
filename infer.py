import numpy as np
import torch
import os
import argparse
import time

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from PIL import Image

import utils
from network import Generator, Discriminator
from dataset import ImgPairDataset

# Global Variables
BATCH_SIZE = 1
IMAGE_SIZE = 256

def infer(args):
    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %s" %torch.cuda.get_device_name(args.gpu))

    # define networks
    g_AtoB = Generator().type(dtype)
    g_BtoA = Generator().type(dtype)

    # load pretrained model parameters
    g_AtoB.load_state_dict(torch.load(args.modelAtoB))
    g_BtoA.load_state_dict(torch.load(args.modelBtoA))

    # set to evaluation mode
    g_AtoB.eval()
    g_BtoA.eval()

    # get training data
    dataset_transform = transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.2), Image.BICUBIC),        # scale shortest side to image_size
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),                # random center image_size out
        transforms.ToTensor(),                                          # turn image from [0-255] to [0-1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # normalize
    ])
    dataloader = DataLoader(ImgPairDataset(args.dataroot, dataset_transform, 'test'), 
                           batch_size = BATCH_SIZE, 
                           shuffle=True)

    # make folders to save data
    A_inStyleOfB_folder = args.dataroot + '/testA_inStyleOfB'
    A_folder = args.dataroot + '/testA_before'
    B_inStyleOfA_folder = args.dataroot + '/testB_inStyleOfA'
    B_folder = args.dataroot + '/testB_before'
    if not os.path.exists(A_inStyleOfB_folder):
        os.makedirs(A_inStyleOfB_folder)
    if not os.path.exists(A_folder):
        os.makedirs(A_folder)
    if not os.path.exists(B_inStyleOfA_folder):
        os.makedirs(B_inStyleOfA_folder)
    if not os.path.exists(B_folder):
        os.makedirs(B_folder)

    # iterate through folder
    for idx, batch in enumerate(dataloader):
        real_A = batch['A'].type(dtype)
        real_B = batch['B'].type(dtype)

        start = time.time()
        A_inStyleOfB = g_AtoB(real_A).cpu()
        end = time.time()
        B_inStyleOfA = g_BtoA(real_B).cpu()

        time_array.append(end - start)

        A_after_imgPath = A_inStyleOfB_folder + '/%03d.png' % (idx)
        B_after_imgPath = B_inStyleOfA_folder + '/%03d.png' % (idx)
        utils.save_image(A_after_imgPath, A_inStyleOfB.data[0])
        utils.save_image(B_after_imgPath, B_inStyleOfA.data[0])

        A_before_imgPath = A_folder + '/%03d.png' % (idx)
        B_before_imgPath = B_folder + '/%03d.png' % (idx)
        utils.save_image(A_before_imgPath, real_A.cpu().data[0])
        utils.save_image(B_before_imgPath, real_B.cpu().data[0])

def main():
    parser = argparse.ArgumentParser(description='Apply CycleGAN with trained models onto a folder of images')

    parser.add_argument("--dataroot", type=str, required=True, help="path to dataset in defined file hierarchy")
    parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    parser.add_argument("--modelAtoB", type=str, required=True, help="path to folder with models for A to B generator")
    parser.add_argument("--modelBtoA", type=str, required=True, help="path to folder with models for B to A generator")

    args = parser.parse_args()

    infer(args)

if __name__ == '__main__':
    main()

