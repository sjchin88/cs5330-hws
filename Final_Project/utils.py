"""
Class Name    : CS5330 Pattern Recognition and Computer Vision
Session       : Fall 2023 (Seattle)
Name          : Shiang Jin Chin
Last Update   : 12/14/2023
Description   : All the utilities function required for the final project
"""

# import statements
import sys
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import models as model


def get_default_device():
    """Return the best device available, cuda if present

    Returns:
        torch.device: default device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def convert_to_device(object, device):
    """convert the object to fit particular device

    Args:
        object(dataloader / model): object to be converted
        device (torch.device): current device used

    Returns:
        dataloader / model: converted object
    """
    if isinstance(object, (list, tuple)):
        return [convert_to_device(x, device) for x in object]
    return object.to(device, non_blocking=True)


def show_example(loader, labels_dict, grayscale=True):
    """
    Display the first 6 examples

    Args:
        loader (torch.utils.data.DataLoader): loader for data set
        labels_dict (dict): contain the mapping of index loaded by torch and the corresponding class name
        grayscale (bool, optional): if required to convert the image to grayscale for display, default is True
    """
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        if grayscale:
            plt.imshow(example_data[i][0], cmap="gray", interpolation='none')
        else:
            plt.imshow(example_data[i].permute(1, 2, 0))
        plt.title(
            f"Species: {labels_dict[example_targets[i].tolist()]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()
    return None


def get_parser():
    """Helper function to build and return the parser for this program
    """
    parser = argparse.ArgumentParser(
        description="Process Command Line Arguments")
    parser.add_argument(
        '-save', help='full absolute path of save directory for data')
    parser.add_argument(
        "-dir_model", help='full absolute path of save directory for model')
    parser.add_argument(
        '-nepochs', help='number of epochs', type=int, default=100)
    parser.add_argument(
        '-train_size', help='batch size for training set', type=int,  default=200)
    parser.add_argument(
        '-test_size', help='batch size for test set', type=int,  default=1000)
    parser.add_argument(
        '-lrate', help='learning rate', type=float, default=0.01)
    parser.add_argument(
        '-momentum', help='momentum', type=float, default=0.5)
    parser.add_argument(
        '-log_int', help='log interval', type=int, default=100)
    parser.add_argument(
        '-test_int', help='test interval', type=int, default=10)
    parser.add_argument(
        '-target_size', help='target size of image for the model', type=int, default=44)
    parser.add_argument(
        '-model', help='selected model, choices of either "2CNN", "3CNN", "Resnet50"', default="2CNN", choices=['2CNN', '3CNN', 'Resnet50'])
    parser.add_argument('--grayscale', action=argparse.BooleanOptionalAction)
    parser.add_argument('--training', action=argparse.BooleanOptionalAction)
    return parser


def plot_performance(train_losses, train_counter, test_losses, test_counter, accuracy_scores, accuracy_scores_top5, test_interval):
    """Plot the performance graph for training losses, test losses, and accuracy scores against number of epochs

    Args:
        train_losses (list): training loss data
        train_counter (list): training counter value
        test_losses (list): test loss data
        test_counter (list): test counter value
        accuracy_scores (list): accuracy score at each test 
        accuracy_scores_top5 (list): accuracy score measured by target in top 5 at each test
        test_interval (int): number of epoch interval between tests
    """
    # Plot the training losses and test losses
    plt.figure(1)
    plt.plot(train_counter, train_losses)
    plt.scatter(test_counter, test_losses)
    plt.legend(['Train Loss', 'Test Loss'], loc='center right')
    plt.xlabel('number of training samples')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    # Plot the accuracy scores against number of epochs
    plt.figure(2)
    nepochs = range(0, len(accuracy_scores)*test_interval, test_interval)
    plt.plot(nepochs, accuracy_scores)
    plt.plot(nepochs, accuracy_scores_top5)
    plt.legend(['Correct guess', 'Top five'], loc='center right')
    plt.xlabel('after n epoch')
    plt.ylabel('accuracy score(%)')
    plt.show()
    return None


def save_data(save_dir, train_losses, train_counter, test_losses, test_counter, accuracy_scores, accuracy_scores_top5, test_interval):
    """Helper function to save the training losses, test losses, and accuracy scores into csv file for record

    Args:
        save_dir (str): files saving directory (must exist)
        train_losses (list): training loss data
        train_counter (list): training counter value
        test_losses (list): test loss data
        test_counter (list): test counter value
        accuracy_scores (list): accuracy score at each test 
        accuracy_scores_top5 (list): accuracy score measured by target in top 5 at each test
        test_interval (int): number of epoch interval between tests
    """
    # Save training loss and display
    df_train = pd.DataFrame(list(zip(train_counter, train_losses)), columns=[
                            'num_samples', 'train_loss'])
    df_train.to_csv(save_dir + "train_loss.csv")
    print(df_train.head(10))

    # Save testing loss and display
    df_test = pd.DataFrame(list(zip(test_counter, test_losses)), columns=[
                           'num_samples', 'test_loss'])
    df_test.to_csv(save_dir + "test_loss.csv")
    print(df_test.head(10))

    # Save the accuracy scores (top1 and top5) and display
    nepochs = range(0, len(accuracy_scores)*test_interval, test_interval)
    df_accuracy = pd.DataFrame(list(zip(nepochs, accuracy_scores, accuracy_scores_top5)), columns=[
                               'num_epoch', 'accuracy_top', 'accuracy_top5'])
    df_accuracy.to_csv(save_dir + "accuracy.csv")
    print(df_accuracy.head(10))
    return None


class CustomDataLoader():
    """Custom class to wrap the data loaders and move the data to the selected device
    """

    def __init__(self, dataloader, device) -> None:
        """Constructor

        Args:
            dataloader (torch.utils.data.DataLoader): dataloader to be converted
            device (torch.device): current device used 
        """
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        """Yield (suspend and return, and available to resume) a batch of data after moving it to the device used
        """
        for databatch in self.dataloader:
            yield convert_to_device(databatch, self.device)

    def __len__(self):
        """Return the length of the dataset owned by this dataloader

        Returns:
            int: length of the dataset
        """
        return len(self.dataloader.dataset)


def load_data(data_save_path, target_size, batch_size_train, batch_size_test, curr_device, grayscale=True, training=True, stored_labels=None):
    """Load the datasets and return the split training and testing set with the label dictionary

    Args:
        data_save_path (str): main directory of the dataset
        target_size (int): target size of the image for model input
        batch_size_train (int): training batch size
        batch_size_test (int): testing batch size
        curr_device (torch.device): current device of this computer
        grayscale (bool, optional): if need to transform image into grayscale, default is True
        training (bool, optional): loading data for training mode (true) or testing mode (false), default is True
        stored_labels (dict, optional): only required to load data for testing mode, default is None

    Returns:
        CustomDataLoader: Data Loader for the training data
        CustomDataLoader: Data Loader for the testing data
        dictionary     : contain the mapping of index loaded by torch and the corresponding class name
    """
    # Load the labels name into a dictionary
    labels_dict = {}
    dir_cnt = 0
    for item in os.listdir(path=data_save_path):
        labels_dict[dir_cnt] = item
        dir_cnt += 1
    print(labels_dict)
    leaf_dataset = None
    if training:
        # Load the images and labels normally
        leaf_dataset = torchvision.datasets.ImageFolder(data_save_path,
                                                        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                                  model.LeafTransform(
                                                            target_size, grayscale),
                                                            torchvision.transforms.Normalize(
                                                            (0.1307,), (0.3081,))]))
    else:
        # we need to modify the target using the species name to suit the actual target value used for training
        labels2idx = {v: k for k, v in stored_labels.items()}
        target_map = {k: labels2idx[v] for k, v in labels_dict.items()}
        leaf_dataset = torchvision.datasets.ImageFolder(data_save_path,
                                                        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                                  model.LeafTransform(
                                                            target_size, grayscale),
                                                            torchvision.transforms.Normalize(
                                                            (0.1307,), (0.3081,))]),
                                                        target_transform=lambda x: target_map[x]
                                                        )
        print(leaf_dataset.targets)
    generator = torch.Generator().manual_seed(42)

    # Set split ratio (test, train) to 0.2: 0.8 for training and
    # 1.0 to 0 for testing mode
    split_ratio = [0.2, 0.8]
    if not training:
        split_ratio = [1.0, 0.0]

    # Split to test and train set
    leaf_dataset_split = torch.utils.data.random_split(
        leaf_dataset, split_ratio, generator=generator)
    leaf_test_data = leaf_dataset_split[0]
    leaf_train_data = leaf_dataset_split[1]
    leaf_train = None
    if training:
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

    # Show some example from the data loader
    if training:
        show_example(leaf_train, labels_dict, grayscale)
        show_example(leaf_test, labels_dict, grayscale)

    # Wrap the train loader for current device
    train_loader = None
    if training:
        train_loader = CustomDataLoader(leaf_train, curr_device)
    test_loader = CustomDataLoader(leaf_test, curr_device)

    return train_loader, test_loader, labels_dict
