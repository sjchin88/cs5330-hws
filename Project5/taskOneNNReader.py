"""
Class Name    : CS5330 Pattern Recognition and Computer Vision
Session       : Fall 2023 (Seattle)
Name          : Shiang Jin Chin
Last Update   : 11/21/2023
Description   : python file containing code required for the first task to read the model
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
from taskOne import MyNetwork

# Parser function, change the default setting here


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
        '-test_size', help='batch size for test set', type=int,  default=10)
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
    batch_size_test = args.test_size

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
    trained_network.eval()

    # Load the test data set
    data_save_path = save_path + "/files/"
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_save_path, train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])), batch_size=batch_size_test, shuffle=False)
    test_samples = enumerate(test_loader)

    # Get the predicted result for the first 10
    # and plot the first 9 examples
    batch_idx, (test_data, test_target) = next(test_samples)
    plt.figure()
    for i in range(10):
        output = trained_network(test_data[i])
        # output.data is the tensor, convert to numpy array
        output_nparray = output.data.numpy()
        output_array = [round(elem, 2) for elem in output_nparray[0]]
        pred = output.data.max(1, keepdim=True)[1][0][0]
        print(
            f"10 output values: {output_array}, max idx: {pred}, correct label: {test_target[i]}")
        if i < 9:
            plt.subplot(3, 3, i+1)
            plt.tight_layout()
            plt.imshow(test_data[i][0], cmap='gray', interpolation='none')
            plt.title("Prediction: {}".format(pred))
            plt.xticks([])
            plt.yticks([])
    plt.show()
    return None


if __name__ == "__main__":
    """default method to run
    """
    main(sys.argv)
