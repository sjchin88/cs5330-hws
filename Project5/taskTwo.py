"""
Class Name    : CS5330 Pattern Recognition and Computer Vision
Session       : Fall 2023 (Seattle)
Name          : Shiang Jin Chin
Last Update   : 11/19/2023
Description   : python file containing code required for the second task to read the model
"""
# import statements
import sys
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
from taskOne import MyNetwork


def get_parser():
    """Helper function to build and return the parser for this program
    """
    parser = argparse.ArgumentParser(
        description="Process Command Line Arguments")
    parser.add_argument(
        '-save', help='full absolute path of save directory for data and model')
    parser.add_argument(
        '-lrate', help='learning rate', type=float, default=0.01)
    parser.add_argument(
        '-momentum', help='momentum', type=float, default=0.5)
    parser.add_argument(
        '-train_size', help='batch size for training set', type=int,  default=64)
    return parser


def main(argv):
    """
    Main Function, contains the main logic to run the training program
    """
    # handle any command line arguments in argv using parser
    parser = get_parser()
    args = parser.parse_args()

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Get the arguments
    save_path = args.save
    learning_rate = args.lrate
    momentum = args.momentum
    batch_size_train = args.train_size

    # Check if path exist
    result_save_path = save_path + "/results/"
    if not os.path.exists(result_save_path):
        print("invalid directory")
        return None

    # Initialize the network and optimizer
    trained_network = MyNetwork()
    trained_optimizer = optim.SGD(
        trained_network.parameters(), lr=learning_rate, momentum=momentum)
    network_state_dict = torch.load(result_save_path + "model.pth")
    trained_network.load_state_dict(network_state_dict)
    optimizer_state_dict = torch.load(result_save_path + "optimizer.pth")
    trained_optimizer.load_state_dict(optimizer_state_dict)
    print(trained_network)
    print(trained_network.convolution_1.weight)

    fig = plt.figure()
    with torch.no_grad():
        for i in range(10):
            filter = trained_network.convolution_1.weight[i, 0]
            # output.data is the tensor, convert to numpy array
            plt.subplot(4, 3, i+1)
            plt.tight_layout()
            plt.imshow(filter)
            plt.title("filter: {}".format(i+1))
            plt.xticks([])
            plt.yticks([])
    plt.show()

    # Load the data set
    data_save_path = save_path + "/files/"
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_save_path, train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])), batch_size=batch_size_train, shuffle=True)
    train_samples = enumerate(train_loader)
    batch_idx, (train_data, train_target) = next(train_samples)
    # print(train_data[0][0])
    src_img = train_data[0][0].numpy()
    # print(src_img)
    fig = plt.figure()
    with torch.no_grad():
        for i in range(10):
            filter = trained_network.convolution_1.weight[i, 0].numpy()
            # output.data is the tensor, convert to numpy array
            plt.subplot(4, 3, i+1)
            plt.tight_layout()
            filtered_img = cv2.filter2D(src_img, ddepth=-1, kernel=filter)
            plt.imshow(filtered_img, cmap='gray', interpolation='none')
            plt.title("filter: {}".format(i+1))
            plt.xticks([])
            plt.yticks([])
    plt.show()
    return None


if __name__ == "__main__":
    """default method to run
    """
    main(sys.argv)
