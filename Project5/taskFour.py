"""
Class Name    : CS5330 Pattern Recognition and Computer Vision
Session       : Fall 2023 (Seattle)
Name          : Shiang Jin Chin
Last Update   : 11/22/2023
Description   : python file containing code required for the fourth task to train the model
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
import argparse

# Custom class for the neural network model


class MyNetwork(nn.Module):
    """Create a custom neural network 

    Args:
        nn (): Default constructor
    """

    def __init__(self, num_filter_conv1=10, num_filter_conv2=20, num_hidden_node=50):
        """_summary_

        Args:
            num_filter_conv1 (int, optional): number of filters in first convolution layer. Defaults to 10.
            num_filter_conv2 (int, optional): number of filters in second convolution layer. Defaults to 20.
            num_hidden_node (int, optional): number of hidden node between the fully connected layer. Defaults to 50.
        """
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filter_conv1, kernel_size=5)
        self.conv2 = nn.Conv2d(
            num_filter_conv1, num_filter_conv2, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        # Output from previous convolution layer = 16 * num_filter_conv2
        self.cnn_output = num_filter_conv2 * 16
        self.fc1 = nn.Linear(self.cnn_output, num_hidden_node)
        self.fc2 = nn.Linear(num_hidden_node, 10)

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
        # Fully connected layer for 320 x 50 with a ReLU function
        x = self.fc1(x)
        x = F.relu(x)
        # Final fully connected layer with 10 nodes and log_softmax function
        x = self.fc2(x)
        return F.log_softmax(x)

# Function to train the network


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
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*batch_size_train) + ((epoch-1)*sample_limit))

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
    with torch.no_grad():
        # Loop through the test data
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    # Compute test loss and print it out
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return accuracy

# Function to show the example data


def show_example(loader):
    """
    Display the shape and first 6 digit examples

    Args:
        loader (torch.utils.data.DataLoader): loader for data set
    """
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data.shape

    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Example digit: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    return None

# Parser function, change the default setting here


def get_parser():
    """Helper function to build and return the parser for this program
    """
    parser = argparse.ArgumentParser(
        description="Process Command Line Arguments")
    parser.add_argument(
        '-save', help='full absolute path of save directory for data and model')
    parser.add_argument(
        '-nepochs', help='number of epochs', type=int, default=10)
    parser.add_argument(
        '-train_size', help='batch size for training set', type=int,  default=50)
    parser.add_argument(
        '-test_size', help='batch size for test set', type=int,  default=1000)
    parser.add_argument(
        '-lrate', help='learning rate', type=float, default=0.01)
    parser.add_argument(
        '-momentum', help='momentum', type=float, default=0.5)
    parser.add_argument(
        '-log_int', help='log interval', type=int, default=100)
    return parser

# Plot the performance of the training loss and test loss


def plot_performance(train_losses, train_counter, test_losses, test_counter, accuracy_scores):
    """Plot the performance graph for test losses and training losses for one test 

    Args:
        train_losses (list): training loss data
        train_counter (list): training counter value
        test_losses (list): test loss data
        test_counter (list): test counter value
        accuracy_scores (list): accuracy score at end of each epoch
    """
    # Get a color map to be used

    plt.figure(1)
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training samples')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    plt.figure(2)
    nepochs = range(1, len(accuracy_scores) + 1, 1)
    plt.plot(nepochs, accuracy_scores)
    plt.xlabel('after n epoch')
    plt.ylabel('accuracy score(%)')
    plt.show()


def plot_experiment_performance(labels, train_losses, train_counters, test_losses, test_counters, accuracy_scores):
    """Plot the performance graph for test losses and training losses for multiple tests

    Args:
        train_losses (list): training loss data
        train_counter (list): training counter value
        test_losses (list): test loss data
        test_counter (list): test counter value
        accuracy_scores (list): accuracy score at end of each epoch
    """
    # cmap = sns.color_palette(n_colors=len(train_losses)*2)
    # print(cmap)
    plt.figure(1, figsize=(12, 10))
    plt.ylim(top=2.5)
    for i in range(len(train_losses)):
        plt.plot(train_counters[i], train_losses[i],
                 label='Train Loss for ' + labels[i])
        # plt.scatter(test_counters[i], test_losses[i], color=cmap[i*2+1], label='Test Loss for ' + labels[i])
    plt.legend(loc='upper right')
    plt.xlabel('number of training samples')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    plt.figure(2, figsize=(12, 10))
    for i in range(len(accuracy_scores)):
        sample_interval = 500_000 // len(accuracy_scores[i])
        nsamples = range(sample_interval, (len(accuracy_scores[i])
                         * sample_interval) + 1, sample_interval)
        plt.plot(nsamples, accuracy_scores[i],
                 label='Accuracy for ' + labels[i])
    plt.legend(loc='lower right')
    plt.xlabel('after n samples')
    plt.ylabel('accuracy score(%)')
    plt.show()


def experiment_sample_size(training_data, test_data, save_path, batch_size_train, batch_size_test, learning_rate, momentum, log_interval):
    """Run the experiment for variation in sample size

    Args:
        training_data (list): list of training_data
        test_data (list): list of test_data
        save_path (str): save path for the trained model
        batch_size_train (int): size of training_batch
        batch_size_test (int): size of testing_batch
        learning_rate (float): learning rate for the optimizer
        momentum (float): momentum for the optimizer
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

    for sample_limit in [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000]:
        # Randomly select data set based on sample_limit
        training_data_extract = random.choices(training_data, k=sample_limit)
        train_loader = torch.utils.data.DataLoader(
            training_data_extract, batch_size=batch_size_train, shuffle=True)
        test_data_extract = random.choices(test_data, k=sample_limit//5)
        test_loader = torch.utils.data.DataLoader(
            test_data_extract, batch_size=batch_size_test, shuffle=True)

        # Target to reach same number of total training samples
        n_epochs = 500_000 // sample_limit
        # Set save_interval for every 10_000 samples
        save_interval = max(10_000 // sample_limit, 1)

        # Initialize the network, optimizer, and arrays to store train_loss, train_counter
        # test_loss, test counter
        network = MyNetwork()
        optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                              momentum=momentum)
        train_loss = []
        train_counter = []
        test_loss = []

        save_sample_interval = 500_000 // (n_epochs // save_interval)
        test_counter = [
            i * save_sample_interval for i in range(n_epochs // save_interval + 1)]
        accuracy_score = []
        model_name = 'sample_' + str(sample_limit)
        test_network(network, test_loader, test_loss)
        for epoch in range(1, n_epochs + 1):
            train_network(network, optimizer, train_loader, epoch,
                          log_interval, train_loss, train_counter, batch_size_train, sample_limit)

            if epoch % save_interval == 0:
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

    plot_experiment_performance(
        labels, train_losses, train_counters, test_losses, test_counters, accuracy_scores)
    return None


def experiment_batch_size(training_data, test_data, save_path, batch_size_test, learning_rate, momentum, n_epoch, log_interval):
    """Run the experiment for variation in batch size

    Args:
        training_data (list): list of training_data
        test_data (list): list of test_data
        save_path (str): save path for the trained model
        batch_size_test (int): size of testing_batch
        learning_rate (float): learning rate for the optimizer
        momentum (float): momentum for the optimizer
        n_epoch (int): number of epoch to run
        log_interval (int): log interval 

    Returns:
        None
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

    for batch_size in [50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000]:
        train_loader = torch.utils.data.DataLoader(
            training_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size_test, shuffle=True)

        # Initialize the network, optimizer, and arrays to store train_loss, train_counter
        # test_loss, test counter
        network = MyNetwork()
        optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                              momentum=momentum)
        train_loss = []
        train_counter = []
        test_loss = []

        test_counter = [i for i in range(n_epoch+1)]
        accuracy_score = []
        model_name = 'batch_' + str(batch_size)
        test_network(network, test_loader, test_loss)
        for epoch in range(1, n_epoch + 1):
            train_network(network, optimizer, train_loader, epoch,
                          log_interval, train_loss, train_counter, batch_size, 50_000)
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

    plot_experiment_performance(
        labels, train_losses, train_counters, test_losses, test_counters, accuracy_scores)
    return None


def experiment_learning_rate(training_data, test_data, save_path, batch_size_train, batch_size_test,  momentum, n_epoch, log_interval):
    """Run the experiment for variation in learning rate

    Args:
        training_data (list): list of training_data
        test_data (list): list of test_data
        save_path (str): save path for the trained model
        batch_size_train (int): size of training_batch
        batch_size_test (int): size of testing_batch
        momentum (float): momentum for the optimizer
        n_epoch (int): number of epoch to run
        log_interval (int): log interval 

    Returns:
        None
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
    for learning_rate in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
        # Initialize the network, optimizer, and arrays to store train_loss, train_counter
        # test_loss, test counter
        network = MyNetwork()
        optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                              momentum=momentum)
        train_loss = []
        train_counter = []
        test_loss = []
        test_counter = [i for i in range(n_epoch+1)]
        accuracy_score = []
        model_name = 'lr_' + str(learning_rate)
        test_network(network, test_loader, test_loss)
        for epoch in range(1, n_epoch+1):
            train_network(network, optimizer, train_loader, epoch,
                          log_interval, train_loss, train_counter, batch_size_train)
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

    plot_experiment_performance(
        labels, train_losses, train_counters, test_losses, test_counters, accuracy_scores)
    return None


def experiment_momentum(training_data, test_data, save_path, batch_size_train, batch_size_test, learning_rate, n_epoch, log_interval):
    """Run the experiment for variation in momentum

    Args:
        training_data (list): list of training_data
        test_data (list): list of test_data
        save_path (str): save path for the trained model
        batch_size_train (int): size of training_batch
        batch_size_test (int): size of testing_batch
        learning_rate (float): learning rate for the optimizer
        n_epoch (int): number of epoch to run
        log_interval (int): log interval 

    Returns:
        None
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
    for momentum in np.arange(0.0, 1.1, 0.1):
        # Initialize the network, optimizer, and arrays to store train_loss, train_counter
        # test_loss, test counter
        network = MyNetwork()
        optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                              momentum=momentum)
        train_loss = []
        train_counter = []
        test_loss = []
        test_counter = [i for i in range(n_epoch+1)]
        accuracy_score = []

        # as np arrange give float value, round it to two decimal places to eliminate the residuals
        momentum = round(momentum, 2)
        model_name = 'momentum_' + str(momentum)

        test_network(network, test_loader, test_loss)
        for epoch in range(1, n_epoch+1):
            train_network(network, optimizer, train_loader, epoch,
                          log_interval, train_loss, train_counter, batch_size_train)
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

    plot_experiment_performance(
        labels, train_losses, train_counters, test_losses, test_counters, accuracy_scores)
    return None


def experiment_adamlr(training_data, test_data, save_path, batch_size_train, batch_size_test, n_epoch, log_interval):
    """Run the experiment for variation in learning rate

    Args:
        training_data (list): list of training_data
        test_data (list): list of test_data
        save_path (str): save path for the trained model
        batch_size_train (int): size of training_batch
        batch_size_test (int): size of testing_batch
        n_epoch (int): number of epoch to run
        log_interval (int): log interval 

    Returns:
        None
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
    for learning_rate in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
        # Initialize the network, optimizer, and arrays to store train_loss, train_counter
        # test_loss, test counter
        network = MyNetwork()
        optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
        train_loss = []
        train_counter = []
        test_loss = []
        test_counter = [i for i in range(n_epoch+1)]
        accuracy_score = []
        model_name = 'adamWlr_' + str(learning_rate)
        test_network(network, test_loader, test_loss)
        for epoch in range(1, n_epoch+1):
            train_network(network, optimizer, train_loader, epoch,
                          log_interval, train_loss, train_counter, batch_size_train)
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

    plot_experiment_performance(
        labels, train_losses, train_counters, test_losses, test_counters, accuracy_scores)
    return None


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
        plot_experiment_performance(
            labels[start:end], train_losses[start:end], train_counters[start:end], test_losses[start:end], test_counters[start:end], accuracy_scores[start:end])
        start = end
    return None


def main(argv):
    """
    Main Function, contains the main logic to run the training program
    """
    # handle any command line arguments in argv using parser
    parser = get_parser()
    args = parser.parse_args()
    # Retrieved required settings
    save_path = args.save
    n_epoch = args.nepochs
    batch_size_train = args.train_size
    batch_size_test = args.test_size
    learning_rate = args.lrate
    momentum = args.momentum
    log_interval = args.log_int

    # Randomize the torch
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    data_save_path = save_path + "/files/"
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    # Load the training and testing data set
    # Download training data from open datasets.
    training_data = torchvision.datasets.FashionMNIST(
        root=data_save_path,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(
        ), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    )

    # Download test data from open datasets.
    test_data = torchvision.datasets.FashionMNIST(
        root=data_save_path,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(
        ), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    )

    # Experiment for variations in sample_size
    # experiment_sample_size(training_data, test_data, save_path, batch_size_train,
    #                       batch_size_test, learning_rate, momentum, log_interval)

    # Experiment for variations in batch size
    # experiment_batch_size(training_data, test_data, save_path,
    #                      batch_size_test, learning_rate, momentum,n_epoch, log_interval)

    # Experiment for variations in learning rate
    # experiment_learning_rate(training_data, test_data, save_path, batch_size_train,
    #                         batch_size_test, momentum, n_epoch, log_interval)

    # Experiment for variations in momentum
    experiment_momentum(training_data, test_data, save_path,
                        batch_size_train, batch_size_test, learning_rate, n_epoch, log_interval)

    # Experiment for variation in learning rate with AdamW optimizer
    # experiment_adamlr(training_data, test_data, save_path, batch_size_train,
    #                  batch_size_test, n_epoch, log_interval)

    # Experiment for variation in filters and hidden nodes combo
    # experiment_filter_node_combo(training_data, test_data, save_path, batch_size_train,
    #                             batch_size_test, learning_rate, momentum, n_epoch, log_interval)
    return


# Entry point
if __name__ == "__main__":
    """default method to run
    """
    main(sys.argv)
