import random

from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms

import numpy as np

# opens and returns image file as a PIL image (0-255)
def load_image(filename):
    img = Image.open(filename)
    return img

# assumes data comes in batch form (ch, h, w) and is between [-1, 1]
def save_image(filename, data):
    scale = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
    add = np.array([1.0, 1.0, 1.0]).reshape((3, 1, 1))
    img = data.clone().numpy()
    img = (((img + add) * scale).transpose(1, 2, 0)*255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

def learning_rate_decay(initial_lr, curr_epoch, total_epochs, optim):
    # keep the same learning rate for first 100 epochs
    if curr_epoch < (total_epochs / 2):
        return

    # decay the learning rate linearly to 0 over the last 100 epochs
    new_lr = initial_lr * ((total_epochs - curr_epoch - 1) / (total_epochs/2))
    for g in optim.param_groups: 
        g['lr'] = new_lr

# replay buffer of size capacity that returns num_sample
# randomly sampled objects from the buffer
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.idx = 0

    def push(self, val):
        # increase buffer size till capacity
        for sample in val.data:
            sample = torch.unsqueeze(sample, 0)

            if len(self.memory) < self.capacity:
                self.memory.append(None)

            # insert at idx and wrap around
            self.memory[self.idx] = val
            self.idx = (self.idx + 1) % self.capacity

    def sample(self, num_samples):
        retList = []
        samples = random.sample(self.memory, num_samples)
        for item in samples:
            retList.append(item.clone())
        return torch.cat(retList)

    def __len__(self):
        return len(self.memory)
