# import statements
import sys
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import random

# Get the utility functions
import utils as util


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
