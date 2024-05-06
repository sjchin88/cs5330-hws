"""
Class Name    : CS5330 Pattern Recognition and Computer Vision
Session       : Fall 2023 (Seattle)
Name          : Shiang Jin Chin
Last Update   : 11/22/2023
Description   : python file containing code required for the first task to read the model, 
                and classify new images
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


# class to transform the new images


class ImgTransform:
    """class to transform the new images
    """

    def __init__(self):
        """Default constructor
        """
        pass

    def __call__(self, x):
        """Operations when called

        Args:
            x (array): representation of input image

        Returns:
            array: representation of output image
        """
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        # Resize to 128 x 128
        x = torchvision.transforms.functional.resize(x, (128, 128))
        # Move it to center, resize to 36 x 36 images
        # x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        # Resize to 28 x 28
        x = torchvision.transforms.functional.resize(x, (28, 28))
        # Return the inverted intensity
        return torchvision.transforms.functional.invert(x)


def get_parser():
    """Helper function to build and return the parser for this program
    """
    parser = argparse.ArgumentParser(
        description="Process Command Line Arguments")
    parser.add_argument(
        '-save', help='full absolute path of save directory for data and model')
    parser.add_argument(
        '-input', help='full absolute path of input directory for image')
    parser.add_argument(
        '-lrate', help='learning rate', type=float, default=0.01)
    parser.add_argument(
        '-momentum', help='momentum', type=float, default=0.5)
    parser.add_argument(
        '-test_size', help='batch size for test set', type=int,  default=9)
    return parser


def evaluate(input_loader, trained_network):
    """Evaluate the performance of the network using custom inputs loaded from input loader

    Args:
        input_loader (torch.utils.data.DataLoader): loader for custom input
        trained_network (torch.nn): the trained nn network
    """
    correct = 0
    count = 0
    for batch_idx, (test_data, test_target) in enumerate(input_loader):

        fig = plt.figure()
        for i in range(min(9, len(test_data))):
            output = trained_network(test_data[i])
            # output.data is the tensor, convert to numpy array
            output_nparray = output.data.numpy()
            output_array = [round(elem, 2) for elem in output_nparray[0]]
            pred = output.data.max(1, keepdim=True)[1][0][0]
            count += 1
            correct += pred.eq(test_target[i])
            # Print output to the console
            print(
                f"10 output values: {output_array}, max idx: {pred}, correct label: {test_target[i]}")
            # Plot the graph for each batch
            if i < 9:
                plt.subplot(3, 3, i+1)
                plt.tight_layout()
                plt.imshow(test_data[i][0], cmap='gray', interpolation='none')
                plt.title("Prediction: {}".format(pred))
                plt.xticks([])
                plt.yticks([])
        plt.show()
    accuracy = correct * 100. / count
    print(f'performance: {correct} / {count}, accuracy score(%) : {accuracy}')
    return None


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
    input_path = args.input
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

    # Load the data set
    input_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(input_path,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   ImgTransform(),
                                                                                   torchvision.transforms.Normalize(
                                             (0.1307,), (0.3081,))])), batch_size=batch_size_test, shuffle=True)

    # Evaluate the performance
    evaluate(input_loader, trained_network)
    return None


if __name__ == "__main__":
    """default method to run
    """
    main(sys.argv)
