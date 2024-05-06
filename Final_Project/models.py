"""
Class Name    : CS5330 Pattern Recognition and Computer Vision
Session       : Fall 2023 (Seattle)
Name          : Shiang Jin Chin
Last Update   : 12/13/2023
Description   : Contains the custom models, leaf_transform class, and other function to train & test the model
"""

# import statements
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time as time
import math


class CNN2Network(nn.Module):
    """Create a custom neural network of two convolution stacks

    Args:
        nn (): Default constructor
    """

    def __init__(self, square_size, num_filter_conv1=10, num_filter_conv2=20, grayscale=False, dynamic_node=False, num_hidden_node=320, num_output=185):
        """_summary_

        Args:
            square_size (int): size of the image to be processed
            num_filter_conv1 (int, optional): number of filters in first convolution layer. Defaults to 10.
            num_filter_conv2 (int, optional): number of filters in second convolution layer. Defaults to 20.
            grayscale (bool, optional): if the input image is in grayscale, default is True
            dynamic_node (bool, optional): if need to set the num of hidden node dynamically. Default is False
            num_hidden_node (int, optional): number of hidden node between the fully connected layer. Defaults to 320.
            num_output (int, optional): number of classification node, defaults to 185
        """
        super(CNN2Network, self).__init__()
        # Set up the convolution layers
        num_input_channel = 3
        if grayscale:
            num_input_channel = 1
        self.conv1 = nn.Conv2d(
            num_input_channel, num_filter_conv1, kernel_size=5)
        self.conv2 = nn.Conv2d(
            num_filter_conv1, num_filter_conv2, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)

        # Set up the fully connected layer, need to get the final size of the image
        # then calculate the number of output from the CNN layers
        final_size = ((square_size - 4)//2 - 4)//2
        self.cnn_output = num_filter_conv2 * final_size * final_size
        if dynamic_node:
            # Optain the ratio between the output from CNN layers with the number of classifications
            # To calculate the number of hidden node
            ratio = math.sqrt(self.cnn_output/num_output)
            num_hidden_node = round(self.cnn_output/ratio)
        self.fc1 = nn.Linear(self.cnn_output, num_hidden_node)
        self.fc2 = nn.Linear(num_hidden_node, num_output)

    def forward(self, x):
        """computes a forward pass for the network
        Args:
            x (data): the data used

        Returns:
            classification results: for the data
        """
        # First convolution layer with a max pooling layer of 2x2 and a ReLU function applied
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Second convolution layer, followed by the dropout layer with 0.5 dropout rate
        # with a max pooling layer of 2x2 and a ReLU function applied
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flattening operation
        x = x.view(-1, self.cnn_output)

        # Fully connected layer with a ReLU function
        x = self.fc1(x)
        x = F.relu(x)
        # Final fully connected layer with log_softmax function
        x = self.fc2(x)
        return F.log_softmax(x)


class CNN3Network(nn.Module):
    """Create a custom neural network of three convolution stacks

    Args:
        nn (): Default constructor
    """

    def __init__(self, square_size, num_filter_conv1=5, num_filter_conv2=10, num_filter_conv3=20, grayscale=False, dynamic_node=False, num_hidden_node=320, num_output=185):
        """_summary_

        Args:
            square_size (int): size of the image to be processed
            num_filter_conv1 (int, optional): number of filters in first convolution layer. Defaults to 10.
            num_filter_conv2 (int, optional): number of filters in second convolution layer. Defaults to 20.
            num_hidden_node (int, optional): number of hidden node between the fully connected layer. Defaults to 50.
            num_output (int, optional): number of classification node, defaults to 50
        """
        super(CNN3Network, self).__init__()

        num_input_channel = 3
        if grayscale:
            num_input_channel = 1
        self.conv1 = nn.Conv2d(
            num_input_channel, num_filter_conv1, kernel_size=5)
        self.conv2 = nn.Conv2d(
            num_filter_conv1, num_filter_conv2, kernel_size=5)
        self.conv3 = nn.Conv2d(
            num_filter_conv2, num_filter_conv3, kernel_size=5)
        self.conv3_drop = nn.Dropout2d(p=0.5)
        # Output from previous convolution layer
        final_size = (((square_size - 4)//2 - 4)//2 - 4)//2
        # print(final_size)
        self.cnn_output = num_filter_conv3 * final_size * final_size
        if dynamic_node:
            # Optain the ratio between the output from CNN layers with the number of classifications
            # To calculate the number of hidden node
            ratio = math.sqrt(self.cnn_output/num_output)
            num_hidden_node = round(self.cnn_output/ratio)
        self.fc1 = nn.Linear(self.cnn_output, num_hidden_node)
        self.fc2 = nn.Linear(num_hidden_node, num_output)

    def forward(self, x):
        """computes a forward pass for the network
        Args:
            x (data): the data used

        Returns:
            classification results: for the data
        """
        # First and second convolution layer with a max pooling layer of 2x2 and a ReLU function applied
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # Third convolution layer, followed by the dropout layer with 0.5 dropout rate
        # with a max pooling layer of 2x2 and a ReLU function applied
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        # Flattening operation
        x = x.view(-1, self.cnn_output)

        # Fully connected layer with a ReLU function
        x = self.fc1(x)
        x = F.relu(x)
        # Final fully connected layer with log_softmax function
        x = self.fc2(x)
        return F.log_softmax(x)


def train_network(network, optimizer, train_loader, epoch, log_interval, train_losses, train_counter,  batch_size_train):
    """Function to train the network

    Args:
        network (torch.nn): the customize neural network
        optimizer (torch.optim): the customize torch optimizer
        train_loader (torch.utils.data.DataLoader): data loader for the training data set
        epoch (int): current number of epoch run
        log_interval (int): number of sample passed for logging
        train_losses (list): List to store the training losses
        train_counter (list): List to store the training counter
        batch_size_train (int): batch size for each training batch
    """
    # Set to training mode
    network.train()

    # Loop through each batch
    for batch_idx, (data, target) in enumerate(train_loader):

        # Reset gradient
        optimizer.zero_grad()
        # Get output
        output = network(data)
        # Calculate loss using nll_loss (negative log-likelihood loss function)
        # nll used for multiclassification problem with softmax layer https://neptune.ai/blog/pytorch-loss-functions
        loss = F.nll_loss(output, target)
        # Compute the gradient and update the parameter using step() function
        loss.backward()
        optimizer.step()

        # Log the training data
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader)))

    return None

# Function to test the network


def test_network(network, test_loader, test_losses):
    """Evaluate performance of the network trained

    Args:
        network (torch.nn): the trained neural network
        test_loader (torch.utils.data.DataLoader): data loader for the test data set
        test_losses (list): List to store the test losses

    returns:
        int: top1 accuracy score 
        int: top5 accuracy score
    """
    # Set to evaluation mode & initialize the recording variables
    network.eval()
    test_loss = 0
    correct = 0
    top5 = 0
    start = time.time()
    with torch.no_grad():
        # Loop through the test data
        for data, target in test_loader:
            output = network(data)
            print(target)
            # Get test loss, prediction, and top 5 results
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            vals, indexes = torch.topk(output, 5)

            # target is a list of target index
            # check if the target within the top 5 results returned
            for i, t in enumerate(target):
                if t in indexes[i]:
                    top5 += 1
    end = time.time()
    # Compute test loss, accuracy scores and print it out
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    accuracy = 100. * correct / len(test_loader)
    accuracy_top5 = 100. * top5 / len(test_loader)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Accuracy in Top5: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader), accuracy, top5, len(test_loader), accuracy_top5))
    print(
        f'\nTotal time taken to evaluate {(len(test_loader)):.4f} samples = {end - start}s, average time taken for each sample = {((end-start)/len(test_loader)):.4f}s')
    return accuracy, accuracy_top5


class LeafTransform:
    """class to transform the new images
    """

    def __init__(self, target_size, grayscale=True, invert=False):
        """Default Constructor

        Args:
            target_size (int): target_size of the output image
            grayscale (bool, optional): if required to transform the image to grayscale, default is True
            invert (bool, optional): if required to invert the image, default is False
        """
        self.target_size = target_size
        self.grayscale = grayscale
        self.invert = invert
        pass

    def __call__(self, x):
        """Operations when called

        Args:
            x (array): representation of input image

        Returns:
            array: representation of output image
        """
        if self.grayscale:
            x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.resize(
            x, (self.target_size, self.target_size))
        if self.invert:
            x = torchvision.transforms.functional.invert(x)
        return x
