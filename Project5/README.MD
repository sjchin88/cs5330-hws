CS5530 Fall 2023 (Seattle) Project 5 Readme

Name: Chin Shiang Jin

Student ID: 002798503

1. Links/URLs to any videos you created and want to submit as part of your report.

Link to handwritten digits and greek letter used 

https://drive.google.com/drive/folders/1NSiGDyd_mue7i_0VHbbizyfzDHc1c554?usp=sharing

2. What operating system and IDE you used to run and compile your code.

Operating system - Window

IDE - Visual Studio Code

3. Instructions for running your executables.
Set Up OS / IDE
- Download and install the Python Kernel
- Use terminal to install require libraries (matplotlib, torch, torchvision) using PIP command
- Refer https://pytorch.org/get-started/locally/ for further details
- Places all py file in the same folder, note some .py file will import class / function developed in other .py file
- For visual studio code, you can add the args in launch json file


4. Instructions for using the program. 

4.1 taskOne.py

The commandline arguments required are

taskOne.py -save <directory>

Available options are as follow

-save <directory> 	full absolute path of intended save directory for data and model (required)

-nepochs <number> 	string representation of number of epochs to run for the training model (integer, optional), default is 5 

-train_size <size>	string representation of batch size for the training set (integer, optional), default is 64

-test_size <size> 	string representation of test size for the training set (integer, optional), default is 1000

-lrate 	<rate>		string representation of learning rate for the model during training (float, optional), default is 0.01

-momentum <rate>	string representation of the momentum for the model during training (float, optional), default is 0.5

-log_int <interval>	string representation of the log interval for the model during training (integer, optional), default is 10


4.2 taskOneNNReader.py

This program perform one simple function, run the saved model and test against random first 10 test samples

The commandline arguments required are

taskOneNNReader.py -save <directory>

Available options are as follow

-save <directory> 	full absolute path of save directory for data and model (required)

-test_size <size> 	string representation of test size for the training set (integer, optional), default is 1000

-lrate 	<rate>		string representation of learning rate for the model during training (float, optional), default is 0.01

-momentum <rate>	string representation of the momentum for the model during training (float, optional), default is 0.5


4.3 taskOneNewInput.py

This program perform one simple function, run the saved model and test against new input loaded. 

For the new input images, group them into the same directory, and separate them into respective subfolder with title as the correct label. 
For example, input images contain digit 0 should all be placed under the directory ../input/0

The commandline arguments required are

taskOneNewInput.py -save <directory> -input <directory>

Available options are as follow

-save <directory> 	full absolute path of save directory for data and model (required)

-input <directory> 	full absolute path of the new input images directory (required)

-test_size <size> 	string representation of test size for the training set (integer, optional), default is 9

-lrate 	<rate>		string representation of learning rate for the model during training (float, optional), default is 0.01

-momentum <rate>	string representation of the momentum for the model during training (float, optional), default is 0.5


4.4 taskTwo.py

This program display the illustration of the 10 filters for the first layer of the network, and the resulting images after applying the filters separately.

The commandline arguments required are

taskTwo.py -save <directory>

Available options are as follow

-save <directory> 	full absolute path of save directory for data and model (required)

-train_size <size> 	string representation of train size for the training set (integer, optional), default is 64

-lrate 	<rate>		string representation of learning rate for the model during training (float, optional), default is 0.01

-momentum <rate>	string representation of the momentum for the model during training (float, optional), default is 0.5


4.5 taskThree.py

This program will transfer the MNIST model trained to classify new images of greek letter, alpha, beta and gamma

The commandline arguments required are

taskThree.py -model_dir <directory>

Available options are as follow

-model_dir <directory> 	full absolute path of save directory for model (required)

-input <directory> 	full absolute path of save directory for input images used for training (required)

-input_test <directory> 	full absolute path of save directory for input images used for test (required)

-nepochs <number> 	string representation of number of epochs to run for the training model (integer, optional), default is 10000 

-train_size <size>	string representation of batch size for the training set (integer, optional), default is 5

-test_size <size> 	string representation of test size for the training set (integer, optional), default is 9

-lrate 	<rate>		string representation of learning rate for the model during training (float, optional), default is 0.01

-momentum <rate>	string representation of the momentum for the model during training (float, optional), default is 0.5

-log_int <interval>	string representation of the log interval for the model during training (integer, optional), default is 10


4.6 taskFour.py

This program will run the experiment testing against different variations of the model. To run a particular test set, just uncomment the 
corresponding lines within the main function to call the corresponding function. Comment out the rest of the function call for other test cases
to avoid performing unrelated tests.

The commandline arguments required are

taskFour.py -save <directory>

Available options are as follow

-save <directory> 	full absolute path of intended save directory for data and model (required)

-nepochs <number> 	string representation of number of epochs to run for the training model (integer, optional), default is 5 

-train_size <size>	string representation of batch size for the training set (integer, optional), default is 50

-test_size <size> 	string representation of test size for the training set (integer, optional), default is 1000

-lrate 	<rate>		string representation of learning rate for the model during training (float, optional), default is 0.01

-momentum <rate>	string representation of the momentum for the model during training (float, optional), default is 0.5

-log_int <interval>	string representation of the log interval for the model during training (integer, optional), default is 100


5. Instructions for testing extensions 

The extension for task 3 is reside in the taskThreeExt.py program. It can be run independently. 

The commandline arguments required are

taskThreeExt.py -save <directory>

Available options are as follow

-model_dir <directory> 	full absolute path of save directory for model (required)

-input <directory> 	full absolute path of save directory for input images used for training (required)

-input_test <directory> 	full absolute path of save directory for input images used for test (required)

-nepochs <number> 	string representation of number of epochs to run for the training model (integer, optional), default is 1000 

-train_size <size>	string representation of batch size for the training set (integer, optional), default is 5

-test_size <size> 	string representation of test size for the training set (integer, optional), default is 9

-lrate 	<rate>		string representation of learning rate for the model during training (float, optional), default is 0.01

-momentum <rate>	string representation of the momentum for the model during training (float, optional), default is 0.5

-log_int <interval>	string representation of the log interval for the model during training (integer, optional), default is 10


6. Time travel days used. 
3 days 