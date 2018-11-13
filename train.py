import numpy as np
import torch
import os
import argparse
import time

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from PIL import Image

import utils
from network import Generator, Discriminator
from dataset import ImgPairDataset
from logger import Logger

# Global Variables
CYCLIC_WEIGHT = 10
BATCH_SIZE = 1
INITIAL_LR = 2e-4
EPOCHS = 200
IMAGE_SIZE = 256
NUM_TEST_SAMPLES = 4

def train(args):
    # set the logger
    logger = Logger('./logs')

    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %s" %torch.cuda.get_device_name(args.gpu))

    # define networks
    g_AtoB = Generator().type(dtype)
    g_BtoA = Generator().type(dtype)
    d_A = Discriminator().type(dtype)
    d_B = Discriminator().type(dtype)

    # optimizers
    optimizer_generators = Adam(list(g_AtoB.parameters()) + list(g_BtoA.parameters()), INITIAL_LR) 
    optimizer_d_A = Adam(d_A.parameters(), INITIAL_LR) 
    optimizer_d_B = Adam(d_B.parameters(), INITIAL_LR)

    # loss criterion
    criterion_mse = torch.nn.MSELoss()
    criterion_l1 = torch.nn.L1Loss()

    # get training data
    dataset_transform = transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1), Image.BICUBIC),        # scale shortest side to image_size
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),                # random center image_size out
        transforms.ToTensor(),                                          # turn image from [0-255] to [0-1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # normalize
    ])
    dataloader = DataLoader(ImgPairDataset(args.dataroot, dataset_transform, 'train'), 
                           batch_size = BATCH_SIZE, 
                           shuffle=True)

    # get some test data to display periodically
    test_data_A = torch.tensor([]).type(dtype)
    test_data_B = torch.tensor([]).type(dtype)
    for i in range(NUM_TEST_SAMPLES):
        imgA = ImgPairDataset(args.dataroot, dataset_transform, 'test')[i]['A'].type(dtype).unsqueeze(0)
        imgB = ImgPairDataset(args.dataroot, dataset_transform, 'test')[i]['B'].type(dtype).unsqueeze(0)
        test_data_A = torch.cat((test_data_A, imgA), dim=0)
        test_data_B = torch.cat((test_data_B, imgB), dim=0)

        fileStrA = 'visualization/test_%d/%s/' % (i, 'B_inStyleofA')
        fileStrB = 'visualization/test_%d/%s/' % (i, 'A_inStyleofB')
        if not os.path.exists(fileStrA):
            os.makedirs(fileStrA)
        if not os.path.exists(fileStrB):
            os.makedirs(fileStrB)

        fileStrA = 'visualization/test_original_%s_%04d.png' % ('A', i)
        fileStrB = 'visualization/test_original_%s_%04d.png' % ('B', i)
        utils.save_image(fileStrA, ImgPairDataset(args.dataroot, dataset_transform, 'test')[i]['A'].data)
        utils.save_image(fileStrB, ImgPairDataset(args.dataroot, dataset_transform, 'test')[i]['B'].data)

    # replay buffers
    replayBufferA = utils.ReplayBuffer(50)
    replayBufferB = utils.ReplayBuffer(50)

    # training loop
    step = 0
    for e in range(EPOCHS):
        startTime = time.time()
        for idx, batch in enumerate(dataloader):
            real_A = batch['A'].type(dtype)
            real_B = batch['B'].type(dtype)

            # some examples seem to have only 1 color channel instead of 3
            if (real_A.shape[1] != 3):
                continue
            if (real_B.shape[1] != 3):
                continue           

            # -----------------
            #  train generators
            # -----------------
            optimizer_generators.zero_grad()
            utils.learning_rate_decay(INITIAL_LR, e, EPOCHS, optimizer_generators)

            # GAN loss
            fake_A = g_BtoA(real_B)
            disc_fake_A = d_A(fake_A)
            fake_B = g_AtoB(real_A)
            disc_fake_B = d_B(fake_B)

            replayBufferA.push(torch.tensor(fake_A.data))
            replayBufferB.push(torch.tensor(fake_B.data))

            target_real = Variable(torch.ones_like(disc_fake_A)).type(dtype)
            target_fake = Variable(torch.zeros_like(disc_fake_A)).type(dtype)

            loss_gan_AtoB = criterion_mse(disc_fake_B, target_real)
            loss_gan_BtoA = criterion_mse(disc_fake_A, target_real)
            loss_gan = loss_gan_AtoB + loss_gan_BtoA

            # cyclic reconstruction loss
            cyclic_A = g_BtoA(fake_B)
            cyclic_B = g_AtoB(fake_A)
            loss_cyclic_AtoBtoA = criterion_l1(cyclic_A, real_A) * CYCLIC_WEIGHT
            loss_cyclic_BtoAtoB = criterion_l1(cyclic_B, real_B) * CYCLIC_WEIGHT
            loss_cyclic = loss_cyclic_AtoBtoA + loss_cyclic_BtoAtoB

            # identity loss
            loss_identity = 0
            loss_identity_A = 0
            loss_identity_B = 0
            if (args.use_identity == True):
                identity_A = g_BtoA(real_A)
                identity_B = g_AtoB(real_B)
                loss_identity_A = criterion_l1(identity_A, real_A) * 0.5 * CYCLIC_WEIGHT
                loss_identity_B = criterion_l1(identity_B, real_B) * 0.5 * CYCLIC_WEIGHT
                loss_identity = loss_identity_A + loss_identity_B

            loss_generators = loss_gan + loss_cyclic + loss_identity
            loss_generators.backward()
            optimizer_generators.step()

            # -----------------
            #  train discriminators
            # -----------------
            optimizer_d_A.zero_grad()
            utils.learning_rate_decay(INITIAL_LR, e, EPOCHS, optimizer_d_A)

            fake_A = replayBufferA.sample(1).detach()
            disc_fake_A = d_A(fake_A)
            disc_real_A = d_A(real_A)
            loss_d_A = 0.5 * (criterion_mse(disc_real_A, target_real) + criterion_mse(disc_fake_A, target_fake))

            loss_d_A.backward()
            optimizer_d_A.step()

            optimizer_d_B.zero_grad()
            utils.learning_rate_decay(INITIAL_LR, e, EPOCHS, optimizer_d_B)

            fake_B = replayBufferB.sample(1).detach()
            disc_fake_B = d_B(fake_B)
            disc_real_B = d_B(real_B)
            loss_d_B = 0.5 * (criterion_mse(disc_real_B, target_real) + criterion_mse(disc_fake_B, target_fake))

            loss_d_B.backward()
            optimizer_d_B.step()

            #log info and save sample images
            if ((idx % 250) == 0):
                # eval on some sample images
                g_AtoB.eval()
                g_BtoA.eval()

                test_B_hat = g_AtoB(test_data_A).cpu()
                test_A_hat = g_BtoA(test_data_B).cpu()

                fileBaseStr = 'test_%d_%d' % (e, idx)
                for i in range(NUM_TEST_SAMPLES):
                    fileStrA = 'visualization/test_%d/%s/%03d_%04d.png' % (i, 'B_inStyleofA', e, idx)
                    fileStrB = 'visualization/test_%d/%s/%03d_%04d.png' % (i, 'A_inStyleofB', e, idx)
                    utils.save_image(fileStrA, test_A_hat[i].data)
                    utils.save_image(fileStrB, test_B_hat[i].data)

                g_AtoB.train()
                g_BtoA.train()

                endTime = time.time()
                timeForIntervalIterations = endTime - startTime
                startTime = endTime

                print('Epoch [{:3d}/{:3d}], Training [{:4d}/{:4d}], Time Spent (s): [{:4.4f}], Losses: [G_GAN: {:4.4f}][G_CYC: {:4.4f}][G_IDT: {:4.4f}][D_A: {:4.4f}][D_B: {:4.4f}]'.format
                    (e, EPOCHS, idx, len(dataloader), timeForIntervalIterations, loss_gan, loss_cyclic, loss_identity, loss_d_A, loss_d_B))

                # tensorboard logging
                info = { 
                    'loss_generators': loss_generators.item(),
                    'loss_gan_AtoB': loss_gan_AtoB.item(),
                    'loss_gan_BtoA': loss_gan_BtoA.item(),
                    'loss_cyclic_AtoBtoA': loss_cyclic_AtoBtoA.item(),
                    'loss_cyclic_BtoAtoB': loss_cyclic_BtoAtoB.item(),
                    'loss_cyclic': loss_cyclic.item(),
                    'loss_d_A': loss_d_A.item(),
                    'loss_d_B': loss_d_B.item(),
                    'lr_optimizer_generators': optimizer_generators.param_groups[0]['lr'],
                    'lr_optimizer_d_A': optimizer_d_A.param_groups[0]['lr'],
                    'lr_optimizer_d_B': optimizer_d_B.param_groups[0]['lr'],
                }
                if (args.use_identity):
                    info['loss_identity_A'] = loss_identity_A.item()
                    info['loss_identity_B'] = loss_identity_B.item()
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)

                info = { 
                    'test_A_hat' : test_A_hat.data.numpy().transpose(0, 2, 3, 1),
                    'test_B_hat' : test_B_hat.data.numpy().transpose(0, 2, 3, 1),
                }
                for tag, images in info.items():
                    logger.image_summary(tag, images, step)

            step += 1

        # save after every epoch
        g_AtoB.eval()
        g_BtoA.eval()
        d_A.eval()
        d_B.eval()

        if use_cuda:
            g_AtoB.cpu()
            g_BtoA.cpu()
            d_A.cpu()
            d_B.cpu()

        if not os.path.exists("models"):
            os.makedirs("models")
        filename_gAtoB = "models/" + str('g_AtoB') + "_epoch_" + str(e) + ".model"
        filename_gBtoA = "models/" + str('g_BtoA') + "_epoch_" + str(e) + ".model"
        filename_dA = "models/" + str('d_A') + "_epoch_" + str(e) + ".model"
        filename_dB = "models/" + str('d_B') + "_epoch_" + str(e) + ".model"
        torch.save(g_AtoB.state_dict(), filename_gAtoB)
        torch.save(g_BtoA.state_dict(), filename_gBtoA)
        torch.save(d_A.state_dict(), filename_dA)
        torch.save(d_B.state_dict(), filename_dB)
        
        if use_cuda:
            g_AtoB.cuda()
            g_BtoA.cuda()
            d_A.cuda()
            d_B.cuda()

def main():
    parser = argparse.ArgumentParser(description='Train CycleGAN')

    parser.add_argument("--dataroot", type=str, required=True, help="path to dataset in defined file hierarchy")
    parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    parser.add_argument("--use-identity", type=int, default=0, help="If the identity loss should be used")

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()

