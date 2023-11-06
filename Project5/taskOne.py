"""
Class Name    : CS5330 Pattern Recognition and Computer Vision
Session       : Fall 2023 (Seattle)
Name          : Shiang Jin Chin
Last Update   : 11/03/2023
Description   : python file containing code required for the first task
"""


# import statements
import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# class definitions


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.convolution_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.convolution_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.pooling = nn.ReLU(nn.MaxPool2d(2))
        self.fully_connect_1 = nn.ReLU(nn.Linear(320, 50))
        self.fully_connect_2 = nn.LogSoftmax(nn.Linear(50, 10))

    # computes a forward pass for the network
    # methods need a summary comment

    def forward(self, x):
        x = self.convolution_1(x)
        x = self.pooling(x)
        x = self.convolution_2(x)
        x = self.conv2_drop(x)
        x = self.pooling(x)
        x = x.view(-1, 320)
        x = self.fully_connect_1(x)
        x = self.fully_connect_2(2)
        return x

# useful functions with a comment for each function


def train_network(arguments):
    return


def show_example(loader):
    """
    Display the shape and first 6 digit examples

    Args:
        loader (): loader for train data
    """
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data.shape

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])

# main function (yes, it needs a comment too)


def main(argv):
    # handle any command line arguments in argv
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])), batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])), batch_size=batch_size_test, shuffle=True)
    show_example(loader=train_loader)
    # main function code
    return


if __name__ == "__main__":
    main(sys.argv)
