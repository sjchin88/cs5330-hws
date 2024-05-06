"""
Class Name    : CS5330 Pattern Recognition and Computer Vision
Session       : Fall 2023 (Seattle)
Name          : Shiang Jin Chin
Last Update   : 11/22/2023
Description   : 
"""

# import statements
import sys
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import time as time


# Get the other custom functions
import utils as util
import experiments as exp


class MyNetwork(nn.Module):
    """Create a custom neural network 

    Args:
        nn (): Default constructor
    """

    def __init__(self, square_size, num_filter_conv1=10, num_filter_conv2=20, num_hidden_node=320, num_output=185):
        """_summary_

        Args:
            square_size (int): size of the image to be processed
            num_filter_conv1 (int, optional): number of filters in first convolution layer. Defaults to 10.
            num_filter_conv2 (int, optional): number of filters in second convolution layer. Defaults to 20.
            num_hidden_node (int, optional): number of hidden node between the fully connected layer. Defaults to 50.
            num_output (int, optional): number of classification node, defaults to 50
        """
        super(MyNetwork, self).__init__()
        final_size = ((square_size - 4)//2 - 4)//2
        self.conv1 = nn.Conv2d(3, num_filter_conv1, kernel_size=5)
        self.conv2 = nn.Conv2d(
            num_filter_conv1, num_filter_conv2, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        # Output from previous convolution layer = 16 * num_filter_conv2
        self.cnn_output = num_filter_conv2 * final_size * final_size
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
        # print(self.cnn_output)
        # print(x)
        x = x.view(-1, self.cnn_output)

        # Fully connected layer for 320 x 50 with a ReLU function
        x = self.fc1(x)
        x = F.relu(x)
        # Final fully connected layer with 10 nodes and log_softmax function
        x = self.fc2(x)
        return F.log_softmax(x)


class LeafTransform:
    """class to transform the new images
    """

    def __init__(self, target_size, invert):
        """Default Constructor

        Args:
            target_size (int): target_size of the output image
            invert (bool): if required to invert the image
        """
        self.target_size = target_size
        self.invert = invert
        pass

    def __call__(self, x):
        """Operations when called

        Args:
            x (array): representation of input image

        Returns:
            array: representation of output image
        """
        # x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.resize(
            x, (self.target_size, self.target_size))
        # x = torchvision.transforms.functional.affine(
        #    x, 0, (0, 0), 44/88, 0)
        if self.invert:
            x = torchvision.transforms.functional.invert(x)
        return x


def train_network(network, optimizer, train_loader, epoch, log_interval, train_losses, train_counter,  batch_size_train, sample_limit=50_000):
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
        sample_limit (int, optional) : number of sample limit (only used for experiment sample), Defaults to 50_000
    """
    # Set to training mode
    network.train()

    training_count = 0
    # Loop through each batch
    for batch_idx, (data, target) in enumerate(train_loader):
        # Accumulate the training count, stop when the count > sample_limit
        training_count += batch_size_train
        if training_count > sample_limit:
            break
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

        # Log the training data and save the interim model
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*batch_size_train) + ((epoch-1)*min(sample_limit, len(train_loader))))

    return None

# Function to test the network


def test_network(network, test_loader, test_losses):
    """Evaluate performance of the network trained

    Args:
        network (torch.nn): the trained neural network
        test_loader (torch.utils.data.DataLoader): data loader for the test data set
        test_losses (list): List to store the test losses
    return 
    """
    # Set to evaluation mode
    network.eval()
    test_loss = 0
    correct = 0
    top5 = 0
    with torch.no_grad():
        # Loop through the test data
        for data, target in test_loader:
            output = network(data)
            # print(output)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            vals, indexes = torch.topk(output, 5)
            # print(indexes)

            for i, t in enumerate(target):
                if t in indexes[i]:
                    top5 += 1

            correct += pred.eq(target.data.view_as(pred)).sum()
    # Compute test loss and print it out
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    accuracy = 100. * correct / len(test_loader)
    accuracy_top5 = 100. * top5 / len(test_loader)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Accuracy in Top5: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader), accuracy, top5, len(test_loader), accuracy_top5))
    return accuracy, accuracy_top5


def experiment_filter_node_combo(training_data, test_data, save_path, batch_size_train, batch_size_test, learning_rate, momentum, n_epoch, log_interval):
    """Run the experiment for variation in number of layers and hidden nodes

    Args:
        training_data (list): list of training_data
        test_data (list): list of test_data
        save_path (str): save path for the trained model
        batch_size_train (int): size of training_batch
        batch_size_test (int): size of testing_batch
        learning_rate (float): learning rate for the optimizer
        momentum (float): momentum for the optimizer
        n_epoch (int): number of epoch to run
        log_interval (int): log interval 
    """
    # We use the list of list to stores the data
    train_losses = []
    train_counters = []
    test_losses = []
    test_counters = []
    accuracy_scores = []
    labels = []
    # create the save path directory if not exist
    result_save_path = save_path + "/results/"
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size_test, shuffle=True)

    for num_filter_conv1 in [5, 10, 15]:
        for filter_ratio in [1, 2, 3]:
            num_filter_conv2 = num_filter_conv1 * filter_ratio
            for ratio in [0.25, 0.5, 0.75]:
                num_hidden_node = round(
                    ((num_filter_conv2 * 16)/10)**ratio * 10)

                # Initialize the network, optimizer, and arrays to store train_loss, train_counter
                # test_loss, test counter
                network = MyNetwork(
                    num_filter_conv1, num_filter_conv2, num_hidden_node)
                optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                      momentum=momentum)
                train_loss = []
                train_counter = []
                test_loss = []

                test_counter = [i for i in range(n_epoch+1)]
                accuracy_score = []
                model_name = 'numcv1_' + str(num_filter_conv1) + '_numcv2_' + str(
                    num_filter_conv2) + '_numNode_' + str(num_hidden_node)
                test_network(network, test_loader, test_loss)
                for epoch in range(1, n_epoch):
                    train_network(network, optimizer, train_loader, epoch,
                                  log_interval, train_loss, train_counter, batch_size_train, 50_000)
                    accuracy = test_network(network, test_loader, test_loss)
                    accuracy_score.append(accuracy)
                    torch.save(network.state_dict(),
                               result_save_path + model_name + '_model.pth')
                    torch.save(optimizer.state_dict(),
                               result_save_path + model_name + '_optimizer.pth')

                # Append the results
                train_losses.append(train_loss)
                train_counters.append(train_counter)
                test_losses.append(test_loss)
                test_counters.append(test_counter)
                accuracy_scores.append(accuracy_score)
                labels.append(model_name)
    start = 0
    for end in range(9, 28, 9):
        exp.plot_experiment_performance(
            labels[start:end], train_losses[start:end], train_counters[start:end], test_losses[start:end], test_counters[start:end], accuracy_scores[start:end])
        start = end
    return None


def main(argv):
    """
    Main Function, contains the main logic to run the training program
    """
    # handle any command line arguments in argv using parser
    parser = util.get_parser()
    args = parser.parse_args()
    # Retrieved required settings
    save_path = args.save
    dir_model = args.dir_model
    n_epoch = args.nepochs
    batch_size_train = args.train_size
    batch_size_test = args.test_size
    learning_rate = args.lrate
    momentum = args.momentum
    log_interval = args.log_int
    test_interval = args.test_int

    # Randomize the torch
    random_seed = 1
    torch.manual_seed(random_seed)

    # Get the current device
    curr_device = util.get_default_device()
    print(curr_device)

    # Load the training data set
    data_save_path = save_path + "/files/"
    if not os.path.exists(data_save_path):
        print('dir invalid')
        return None

    leaf_dataset = torchvision.datasets.ImageFolder(data_save_path,
                                                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                              LeafTransform(
                                                        44, False),
                                                        torchvision.transforms.Normalize(
                                                        (0.1307,), (0.3081,))]))
    generator = torch.Generator().manual_seed(42)
    leaf_dataset_split = torch.utils.data.random_split(
        leaf_dataset, [0.2, 0.8], generator=generator)
    leaf_test_data = leaf_dataset_split[0]
    leaf_train_data = leaf_dataset_split[1]
    leaf_train = torch.utils.data.DataLoader(
        leaf_train_data,
        batch_size=batch_size_train,
        shuffle=True,
    )
    leaf_test = torch.utils.data.DataLoader(
        leaf_test_data,
        batch_size=batch_size_test,
        shuffle=True,
    )

    # Load the labels name into a dictionary
    labels_dict = {}
    dir_cnt = 0
    for item in os.listdir(path=data_save_path):
        labels_dict[dir_cnt] = item
        dir_cnt += 1
    print(labels_dict)

    # Show some example from the data loader
    util.show_example(leaf_train, labels_dict)
    util.show_example(leaf_test, labels_dict)

    # Wrap the train loader for current device
    train_loader = util.CustomDataLoader(leaf_train, curr_device)
    test_loader = util.CustomDataLoader(leaf_test, curr_device)

    # Initialize the network, optimizer, and arrays to store train_loss, train_counter
    # Check if model storage path exist
    result_save_path = dir_model + "/results_2layer_rgb/"
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
        # return None

    # Initialize the network and optimizer
    network = MyNetwork(44, num_filter_conv1=10,
                        num_filter_conv2=20, num_hidden_node=320, num_output=len(labels_dict))
    util.convert_to_device(network, curr_device)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)
    # Initialize arrays required to store the result
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader)
                    for i in range(0, n_epoch + 1, test_interval)]
    accuracy_score = []
    accuracy_score_top5 = []
    # accuracy_counter = [i for i in range(0, n_epoch+1, 10)]
    start = time.time()
    accuracy, accuracy_top5 = test_network(network, test_loader, test_losses)
    accuracy_score.append(accuracy)
    accuracy_score_top5.append(accuracy_top5)
    for epoch in range(1, n_epoch + 1):
        # Train the network
        train_network(network, optimizer, train_loader, epoch,
                      log_interval, train_losses, train_counter, batch_size_train)

        # Test network every 10 epoch
        if epoch % test_interval == 0:
            accuracy, accuracy_top5 = test_network(
                network, test_loader, test_losses)
            accuracy_score.append(accuracy)
            accuracy_score_top5.append(accuracy_top5)
            print(
                f"accuracy for epoch {epoch}: {accuracy} %")
            torch.save(network.state_dict(),
                       result_save_path + 'model_leaf.pth')
            torch.save(optimizer.state_dict(),
                       result_save_path + 'optimizer_leaf.pth')
    end = time.time()
    print(
        f"Total training time: {end - start}s, est training time per epoch = {(end-start)/epoch}s")
    accuracy_score = [score.tolist() for score in accuracy_score]
    # print(accuracy_score)
    # accuracy_score = [score.tolist() for score in accuracy_score]
    # accuracy_score_top5 = [score.tolist() for score in accuracy_score_top5]
    # Print the result and performance of the trained model
    util.plot_performance(train_losses, train_counter, test_losses,
                          test_counter, accuracy_score, accuracy_score_top5, test_interval)
    util.save_data(result_save_path, train_losses, train_counter, test_losses,
                   test_counter, accuracy_score, accuracy_score_top5, test_interval)

    return


# Entry point
if __name__ == "__main__":
    """default method to run
    """
    main(sys.argv)
