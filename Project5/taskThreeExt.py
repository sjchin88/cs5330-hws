"""
Class Name    : CS5330 Pattern Recognition and Computer Vision
Session       : Fall 2023 (Seattle)
Name          : Shiang Jin Chin
Last Update   : 11/22/2023
Description   : python file containing code required for the extension for task 3
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
        x = torchvision.transforms.functional.resize(x, (128, 128))
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def train_network(network, optimizer, train_loader, epoch, log_interval, train_losses, train_counter):
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
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*5) + ((epoch-1)*len(train_loader.dataset)))

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

    return correct


def get_parser():
    """Helper function to build and return the parser for this program
    """
    parser = argparse.ArgumentParser(
        description="Process Command Line Arguments")
    parser.add_argument(
        '-model_dir', help='full absolute path of save directory for model')
    parser.add_argument(
        '-input', help='full absolute path of input directory for images used in training')
    parser.add_argument(
        '-input_test', help='full absolute path of input directory for images used in testing')
    parser.add_argument(
        '-nepochs', help='number of epochs', type=int, default=1000)
    parser.add_argument(
        '-train_size', help='batch size for training set', type=int,  default=5)
    parser.add_argument(
        '-test_size', help='batch size for test set', type=int,  default=9)
    parser.add_argument(
        '-lrate', help='learning rate', type=float, default=0.01)
    parser.add_argument(
        '-momentum', help='momentum', type=float, default=0.5)
    parser.add_argument(
        '-log_int', help='log interval', type=int, default=5)
    return parser


def plot_performance(train_losses, train_counter, test_losses, test_counter):
    """Plot the performance graph for test losses and training losses

    Args:
        train_losses (list): training loss data
        train_counter (list): training counter value
        test_losses (list): test loss data
        test_counter (list): test counter value
    """
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training samples')
    plt.ylabel('negative log likelihood loss')
    plt.show()


def evaluate(input_loader, trained_network):
    """Evaluate the performance of the network using custom inputs loaded from input loader

    Args:
        input_loader (torch.utils.data.DataLoader): loader for custom input
        trained_network (torch.nn): the trained nn network
    """
    correct = 0
    count = 0
    for batch_idx, (test_data, test_target) in enumerate(input_loader):

        plt.figure()
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


def retrain_network(trained_network, trained_optimizer, greek_train, n_epochs, log_interval, train_losses, train_counters, test_losses, test_counters, accuracy_scores, save_name):
    # Initialize arrays required to store the result
    train_loss = []
    train_counter = []
    test_loss = []
    test_counter = [i*len(greek_train.dataset)
                    for i in range(0, n_epochs + 1, 100)]
    accuracy_score = []

    test_network(trained_network, greek_train, test_loss)
    for epoch in range(1, n_epochs + 1):
        # Train the network
        train_network(trained_network, trained_optimizer, greek_train,
                      epoch, log_interval, train_loss, train_counter)

        # Test network every 1000 epoch
        if epoch % 100 == 0:
            correct = test_network(trained_network, greek_train, test_loss)
            print(f"accuracy for epoch {epoch}: {correct} / 27")
            accuracy = 100. * correct / 27
            accuracy_score.append(accuracy)
            torch.save(trained_network.state_dict(),
                       save_name + "model.pth")
            torch.save(trained_optimizer.state_dict(),
                       save_name + "optimizer.pth")

    # Apend the performance of the trained model
    plot_performance(train_loss, train_counter, test_loss, test_counter)
    train_losses.append(train_loss)
    train_counters.append(train_counter)
    test_losses.append(test_loss)
    test_counters.append(test_counter)
    accuracy_scores.append(accuracy_score)


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
        plt.plot(test_counters[i], test_losses[i],
                 label='Test Loss for ' + labels[i])
        # plt.scatter(test_counters[i], test_losses[i], color=cmap[i*2+1], label='Test Loss for ' + labels[i])
    plt.legend(loc='upper right')
    plt.xlabel('number of training samples')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    plt.figure(2, figsize=(12, 10))
    for i in range(len(accuracy_scores)):
        sample_interval = 27_000 // len(accuracy_scores[i])
        nsamples = range(sample_interval, (len(accuracy_scores[i])
                         * sample_interval) + 1, sample_interval)
        plt.plot(nsamples, accuracy_scores[i],
                 label='Accuracy for ' + labels[i])
    plt.legend(loc='lower right')
    plt.xlabel('after n samples')
    plt.ylabel('accuracy score(%)')
    plt.show()


def main(argv):
    """
    Main Function, contains the main logic to run the training program
    """
    # handle any command line arguments in argv using parser
    parser = get_parser()
    args = parser.parse_args()
    # print(args)
    save_path = args.model_dir
    input_path = args.input
    test_input_path = args.input_test
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
    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(input_path,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                             (0.1307,), (0.3081,))])),
        batch_size=batch_size_train,
        shuffle=True)

    # Load the test data set
    greek_test = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(test_input_path,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                             (0.1307,), (0.3081,))])),
        batch_size=batch_size_test,
        shuffle=True)

    # Initialize the network, optimizer, and arrays to store train_loss, train_counter
    # Check if path exist
    result_save_path = save_path + "/results/"
    if not os.path.exists(result_save_path):
        print("invalid directory")
        return None

    # We use the list of list to stores the data
    train_losses = []
    train_counters = []
    test_losses = []
    test_counters = []
    accuracy_scores = []
    labels = []

    # First alternative test, retrain the network from scratch
    # Initialize the network and optimizer
    trained_network = MyNetwork()
    trained_optimizer = optim.SGD(
        trained_network.parameters(), lr=learning_rate, momentum=momentum)

    # Replace the last classification layer
    trained_network.fc2 = nn.Linear(50, 3)

    save_name = result_save_path + "_new_"
    retrain_network(trained_network, trained_optimizer, greek_train, n_epochs, log_interval,
                    train_losses, train_counters, test_losses, test_counters, accuracy_scores, save_name)
    evaluate(greek_test, trained_network)
    labels.append("New Model")

    # Second alternative test, load the model but do not freeze the parameters
    # Initialize the network and optimizer
    trained_network = MyNetwork()
    trained_optimizer = optim.SGD(
        trained_network.parameters(), lr=learning_rate, momentum=momentum)

    # Load the pretrained model
    network_state_dict = torch.load(result_save_path + "model.pth")
    trained_network.load_state_dict(network_state_dict)
    optimizer_state_dict = torch.load(result_save_path + "optimizer.pth")
    trained_optimizer.load_state_dict(optimizer_state_dict)

    # Replace the last classification layer
    trained_network.fc2 = nn.Linear(50, 3)

    save_name = result_save_path + "_nofreeze_"
    retrain_network(trained_network, trained_optimizer, greek_train, n_epochs, log_interval,
                    train_losses, train_counters, test_losses, test_counters, accuracy_scores, save_name)
    evaluate(greek_test, trained_network)
    labels.append("No Freeze Model")

    # Third alternative test, using adamW optimizer
    trained_network = MyNetwork()
    trained_optimizer = optim.AdamW(
        trained_network.parameters(), lr=learning_rate)
    # Load the training and testing data set
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
    # Train new network for 5 epochs
    for epoch in range(1, 6):
        train_network(trained_network, trained_optimizer, train_loader, epoch,
                      log_interval, [], [])

    # freezes the parameters for the whole network
    for param in trained_network.parameters():
        param.requires_grad = False

    # Replace the last classification layer
    trained_network.fc2 = nn.Linear(50, 3)

    save_name = result_save_path + "_nofreeze_"
    retrain_network(trained_network, trained_optimizer, greek_train, n_epochs, log_interval,
                    train_losses, train_counters, test_losses, test_counters, accuracy_scores, save_name)
    evaluate(greek_test, trained_network)
    labels.append("AdamW Model")

    plot_experiment_performance(
        labels, train_losses, train_counters, test_losses, test_counters, accuracy_scores)
    return


if __name__ == "__main__":
    """default method to run
    """
    main(sys.argv)