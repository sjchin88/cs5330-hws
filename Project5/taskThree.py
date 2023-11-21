"""
Class Name    : CS5330 Pattern Recognition and Computer Vision
Session       : Fall 2023 (Seattle)
Name          : Shiang Jin Chin
Last Update   : 11/19/2023
Description   : python file containing code required for the first task to train the model
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


class MyNetwork(nn.Module):
    """Create a custom neural network 

    Args:
        nn (): Default constructor
    """

    def __init__(self):
        """Create a custom neural network 
        """
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """computes a forward pass for the network
        Args:
            x (data): the data used

        Returns:
            classification results: for the data
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x)

# greek data set transform


class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.resize(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def train_network(network, optimizer, save_path, train_loader, epoch, log_interval, train_losses, train_counter):
    """_summary_

    Args:
        network (torch.nn): the customize neural network
        optimizer (torch.optim): the customize torch optimizer
        save_path  (str): save path for the results
        train_loader (torch.utils.data.DataLoader): data loader for the training data set
        epoch (int): current number of epoch run
        log_interval (int): number of sample passed for logging
        train_losses (list): List to store the training losses
        train_counter (list): List to store the training counter
    """
    network.train()
    result_save_path = save_path + "/results/"

    # Make directory of result path
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    # Go through each batch
    for batch_idx, (data, target) in enumerate(train_loader):

        # Get the loss
        output = network(data)
        loss = F.nll_loss(output, target)

        # Propagate backward
        loss.backward()
        optimizer.step()
        # Reset grad to zero
        optimizer.zero_grad()

        # Log interval data
        if batch_idx % log_interval == 0:
            """
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            """
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*5) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(),
                       result_save_path + 'modelGr.pth')
            torch.save(optimizer.state_dict(),
                       result_save_path + 'optimizerGr.pth')
    return None


def test_network(network, test_loader, test_losses):
    """Evaluate performance of the network trained

    Args:
        network (torch.nn): the trained neural network
        test_loader (torch.utils.data.DataLoader): data loader for the test data set
        test_losses (list): List to store the test losses
    """
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    """
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        """
    return correct


def get_parser():
    """Helper function to build and return the parser for this program
    """
    parser = argparse.ArgumentParser(
        description="Process Command Line Arguments")
    parser.add_argument(
        '-save', help='full absolute path of save directory for data and model')
    parser.add_argument(
        '-nepochs', help='number of epochs', type=int, default=200)
    parser.add_argument(
        '-train_size', help='batch size for training set', type=int,  default=64)
    parser.add_argument(
        '-test_size', help='batch size for test set', type=int,  default=1000)
    parser.add_argument(
        '-lrate', help='learning rate', type=float, default=0.01)
    parser.add_argument(
        '-momentum', help='momentum', type=float, default=0.5)
    parser.add_argument(
        '-log_int', help='log interval', type=int, default=5)
    return parser


def show_example(loader):
    """
    Display the shape and first 6 digit examples

    Args:
        loader (torch.utils.data.DataLoader): loader for data set
    """
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data.shape

    fig = plt.figure()
    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Example Greek: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    return None


def plot_performance(train_losses, train_counter, test_losses, test_counter):
    """Plot the performance graph for test losses and training losses

    Args:
        train_losses (list): training loss data
        train_counter (list): training counter value
        test_losses (list): test loss data
        test_counter (list): test counter value
    """
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training samples')
    plt.ylabel('negative log likelihood loss')
    plt.show()


def main(argv):
    """
    Main Function, contains the main logic to run the training program
    """
    # handle any command line arguments in argv using parser
    parser = get_parser()
    args = parser.parse_args()
    # print(args)
    save_path = args.save

    n_epochs = args.nepochs
    batch_size_train = args.train_size
    batch_size_test = args.test_size
    learning_rate = args.lrate
    momentum = args.momentum
    log_interval = args.log_int

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Load the data set
    data_save_path = save_path + "/files/greek_train/"
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
     # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(data_save_path,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                             (0.1307,), (0.3081,))])),
        batch_size=27,
        shuffle=True)
    # show_example(loader=greek_train)

    # Initialize the network, optimizer, and arrays to store train_loss, train_counter
    # Check if path exist
    result_save_path = save_path + "/results/"
    if not os.path.exists(result_save_path):
        print("invalid directory")
        return None

    # Initialize the network and optimizer
    trained_network = MyNetwork()

    # Load the pretrained model
    network_state_dict = torch.load(result_save_path + "model.pth")
    trained_network.load_state_dict(network_state_dict)
    optimizer_state_dict = torch.load(result_save_path + "optimizer.pth")
    trained_optimizer.load_state_dict(optimizer_state_dict)

    # freezes the parameters for the whole network
    for name, param in trained_network.named_parameters():
        # print(param.)
        param.requires_grad = False

    # Replace the last classification layer
    trained_network.fc2 = nn.Linear(50, 3)
    trained_optimizer = optim.SGD(
        trained_network.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(greek_train.dataset)
                    for i in range(0, n_epochs + 1, 100)]

    test_network(trained_network, greek_train, test_losses)
    for epoch in range(1, n_epochs + 1):
        # Train the network
        train_network(trained_network, trained_optimizer, save_path,
                      greek_train, epoch, log_interval, train_losses, train_counter)
        if epoch % 100 == 0:
            # Get test result every 100 loops
            correct = test_network(trained_network, greek_train, test_losses)
            # if correct == len(greek_train.dataset):
            #    break

    plot_performance(train_losses, train_counter, test_losses, test_counter)
    # print(trained_network)
    # print(trained_network.fully_connect_1.weight)
    # print(trained_network.fully_connect_2.weight)
    return


if __name__ == "__main__":
    """default method to run
    """
    main(sys.argv)
