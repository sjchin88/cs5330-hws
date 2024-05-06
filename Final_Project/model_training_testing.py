"""
Class Name    : CS5330 Pattern Recognition and Computer Vision
Session       : Fall 2023 (Seattle)
Name          : Shiang Jin Chin
Last Update   : 12/13/2023
Description   : Code to train the model or test the model
"""

# import statements
import sys
import os
import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time as time
import torch.nn as nn
import collections
import json

# Get the other custom functions
import utils as util
import models as model


def run_training(network, optimizer, result_save_path, train_loader, test_loader, n_epoch, log_interval, batch_size_train, test_interval):
    """Main function to run the training & testing process. 
    Trained model and optimizer will be saved into the result_save_path

    Args:
        network (torch.nn): model for training
        optimizer (torch.optim): optimizer for the model use
        result_save_path (str): save path for the result
        train_loader (CustomDataLoader): DataLoader for the training data
        test_loader (CustomDataLoader): DataLoader for the testing data
        n_epoch (int): number of epochs to run 
        log_interval (int): logging interval for the training
        batch_size_train (int): batch size for each training
        test_interval (int): testing interval (epochs) 

    Returns:
        torch.nn : the trained model
    """
    # Initialize arrays required to store the result
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader)
                    for i in range(0, n_epoch + 1, test_interval)]
    accuracy_score = []
    accuracy_score_top5 = []

    # Start the timer and start counting
    start = time.time()
    accuracy, accuracy_top5 = model.test_network(
        network, test_loader, test_losses)
    accuracy_score.append(accuracy)
    accuracy_score_top5.append(accuracy_top5)
    for epoch in range(1, n_epoch + 1):
        # Train the network
        model.train_network(network, optimizer, train_loader, epoch,
                            log_interval, train_losses, train_counter, batch_size_train)

        # Test network every 10 epoch
        if epoch % test_interval == 0:
            accuracy, accuracy_top5 = model.test_network(
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

    # Print the result and performance of the trained model
    util.plot_performance(train_losses, train_counter, test_losses,
                          test_counter, accuracy_score, accuracy_score_top5, test_interval)
    util.save_data(result_save_path, train_losses, train_counter, test_losses,
                   test_counter, accuracy_score, accuracy_score_top5, test_interval)
    return network


def get_resnet(class_number):
    """Helper function to build the custom resnet model
    Args:
        class_number (int): number of class to be classified

    Returns:
        torchvision.model : the model built
    """
    # Initialize the network and optimizer
    network = torchvision.models.resnet50(weights=True)
    print(network)

    # freeze pretrained model parameters
    for parameter in network.parameters():
        parameter.requires_grad = False

    # Replace the last layer
    print("Original final layer")
    print(network.fc)
    classifier = nn.Sequential(
        collections.OrderedDict(
            [
                ("fc", nn.Linear(network.fc.in_features, class_number)),
                ("out", nn.LogSoftmax(dim=1))
            ]
        )
    )
    network.fc = classifier
    print("\nModified final layer")
    print(network.fc)
    return network


def evaluate(input_loader, trained_network, labels_dict):
    """Evaluate the performance of the network using custom inputs loaded from input loader

    Args:
        input_loader (torch.utils.data.DataLoader): loader for custom input
        trained_network (torch.nn): the trained nn network
        labels_dict (dict) : map the predicted index to the label stored
    """
    trained_network.eval()
    with torch.no_grad():
        for batch_idx, (test_data, test_target) in enumerate(input_loader):
            plt.figure(figsize=(12, 12))
            # Show the sample for the first 9 data only
            output = trained_network(test_data)
            vals, indexes = torch.topk(output, 5)
            for i in range(min(8, len(test_data))):

                # pred = output.data.max(1, keepdim=True)[1][0][0]

                print(f"output indexes{indexes[i]}")
                prediction_txt = "Top5:\n"
                for idx in indexes[i]:
                    prediction_txt += str(idx.tolist()) + \
                        ":" + labels_dict[idx.tolist()] + "\n"
                # Plot the graph
                plt.subplot(4, 4, i*2+1)
                plt.tight_layout()
                test_img = torch.Tensor.cpu(test_data[i])
                plt.imshow(test_img.permute(1, 2, 0), interpolation='none')
                plt.title(
                    f"Target:{labels_dict[test_target[i].tolist()]}")
                plt.xticks([])
                plt.yticks([])
                subfig = plt.subplot(4, 4, i*2+2)
                subfig.text(0.5, 0.5, prediction_txt, horizontalalignment='center',
                            verticalalignment='center', transform=subfig.transAxes)

            plt.show()
            break
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
    target_size = args.target_size
    selected_model = args.model
    grayscale = args.grayscale
    print(f"value of grayscale:{grayscale}")
    training_mode = args.training
    print(f"value of training_mode:{training_mode}")

    # Randomize the torch
    random_seed = 1
    torch.manual_seed(random_seed)

    # Get the current device
    curr_device = util.get_default_device()
    print(f"current device is : {curr_device}")

    # Load the training data set
    data_save_path = save_path
    if not os.path.exists(data_save_path):
        print('dir invalid')
        return None

    # Check if model storage path exist
    result_save_path = dir_model
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
        # return None

    # Loading the data
    stored_labels = None
    if not training_mode:
        # Load the label dict from the json.txt file
        def keys2int(x):
            return {int(k): v for k, v in x}
        with open(result_save_path+"labels.txt") as json_file:
            stored_labels = json.load(json_file, object_pairs_hook=keys2int)
        print(stored_labels)
    class_number = len(stored_labels)

    train_loader, test_loader, labels_dict = util.load_data(
        data_save_path, target_size, batch_size_train, batch_size_test, curr_device, grayscale, training=training_mode, stored_labels=stored_labels)

    if training_mode:
        # Save the label dict together with the model & update the class_number
        with open(result_save_path+"labels.txt", "w") as json_file:
            json.dump(labels_dict, json_file)
        class_number = len(labels_dict)
    else:
        labels_dict = stored_labels

    # Initialize the network and optimizer based on selected model
    network = None
    if selected_model == "2CNN":
        network = model.CNN2Network(target_size, num_filter_conv1=10,
                                    num_filter_conv2=20, grayscale=grayscale, num_output=class_number)
    elif selected_model == "3CNN":
        network = model.CNN3Network(target_size, num_filter_conv1=5,
                                    num_filter_conv2=10, num_filter_conv3=20, grayscale=grayscale, num_hidden_node=320, num_output=class_number)
    elif selected_model == "Resnet50":
        network = get_resnet(class_number)
    else:
        print("no model selected")
    util.convert_to_device(network, curr_device)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    # Train or test the network
    if training_mode:
        # Run the training
        network = run_training(network, optimizer, result_save_path, train_loader,
                               test_loader, n_epoch, log_interval, batch_size_train, test_interval)
    else:
        network_state_dict = torch.load(result_save_path + "model_leaf.pth")
        network.load_state_dict(network_state_dict)
        # util.convert_to_device(network, curr_device)
        optimizer_state_dict = torch.load(
            result_save_path + "optimizer_leaf.pth")
        optimizer.load_state_dict(optimizer_state_dict)
        print(labels_dict)
        evaluate(test_loader, network, labels_dict)
        model.test_network(network, test_loader, [])

    return


# Entry point
if __name__ == "__main__":
    """default method to run
    """
    main(sys.argv)
