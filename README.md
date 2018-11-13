# CycleGAN

## Summary
This project is a PyTorch implementation of [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593). Given images representing two different domains of data (for example, Monet paintings and real world photographs or pictures of horses and zebras), we learn a pair of autoencoders that maps from domain A to domain B and vice versa. Notably compared to prior methods, this paper does not need paired training data. This simple and novel intuition of the paper is by chaining a function that maps from domain A to domain B and another function that maps from domain B to domain A, we should get an output that is similar to the original input.

## Prerequisites
- Python 3.6
- [PyTorch 0.4.0](http://pytorch.org/)
- [NumPy](http://www.numpy.org/)
- [PIL](http://pillow.readthedocs.io/en/3.1.x/installation.html)
- [Tensorboard and TensorFlow](https://www.tensorflow.org) for logging

## Usage
### Train

The code assumes a dataset folder hierarchy below, as used by the original authors [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/):

```bash
datasetAtoB
    |____trainA
    |____testA
    |____trainB
    |____testB
```

To train, run the following command

```bash
$ python train.py --dataroot path_to_dataset_AtoB
```

These are some additional parameters that you can use:
* `--gpu`: (int) id of the GPU you want to use (if not specified, will train on CPU)
* `--use-identity`: (int) 0 or 1; whether the identity loss should be added to the loss function for training the generators (i.e., when given data from the domain the generator should be mapping to, the generator should act as an identity function)

Data need to run TensorBoard will be written out during training at regular intervals. You can start a Tensorboard server from the root of the project directory with:

```bash
$ tensorboard --logdir='./logs' --port 6006
```

I also write out a few images from the test set with the generators applied as another debugging tool. These images appear in the visualization folder.

### Evaluation

Evaluation was set up to apply the generators on all the test images within the dataset hierarchy previously defined. Going through the dataset folders, the predicted output will be written in folders `testA_before`, `testA_inStyleOfB`, `testB_before`, and `testB_inStyleOfA`. Of course, the code can be trivially modified to run on arbitrary directories of data. 

```bash
$ python infer.py --dataroot path_to_dataset_AtoB --modelAtoB model_path_AtoB --modelBtoA model_path_BtoA
```

You can also specify if you would like to run on a GPU:
* `--gpu`: (int) id of the GPU you want to use (if not specified, will train on CPU)

## Results

Training a GAN byitself is already a fickle process, so concurrently training two GANs was not the simplest process. Notably, I'd reach some training scenarios where one determinator would get into a state where it stopped learning (i.e., started guessing 50% likelihood for everything) which broke the information flow as little useful signal was provided to the generator.

Here are some results from horse2zebra. Original horse image on the left and syntheticly stylized as a zebra on the right.

<p align="center">
    <img src="resources/horse2zebra_horse_0.jpg" height="200px">
    <img src="resources/horse2zebra_zebra_0.jpg" height="200px">
</p>
<p align="center">
    <img src="resources/horse2zebra_horse_1.jpg" height="200px">
    <img src="resources/horse2zebra_zebra_1.jpg" height="200px">
</p>
<p align="center">
    <img src="resources/horse2zebra_horse_2.jpg" height="200px">
    <img src="resources/horse2zebra_zebra_2.jpg" height="200px">
</p>

Unfortunately, learning the reverse function was a bit more difficult. Original zebra image on the left and synthetically stlized as a horse on the right.

<p align="center">
    <img src="resources/zebra2horse_zebra_0.jpg" height="200px">
    <img src="resources/zebra2horse_horse_0.jpg" height="200px">
</p>
<p align="center">
    <img src="resources/zebra2horse_zebra_1.jpg" height="200px">
    <img src="resources/zebra2horse_horse_1.jpg" height="200px">
</p>

On a GTX1070, inference can be done in approximately 0.0113 seconds or approximately at a frequency of 88 Hz for a 256 x 256 x 3 image (ignoring time spent loading the image and moving it onto GPU memory). It would be interesting to try applying this in high performance, real-time environments. 
