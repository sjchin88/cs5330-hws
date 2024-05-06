CS5530 Fall 2023 (Seattle) Final Project Readme

Project Description: 

The main part of the project will be building and training the convolution neural network of different architecture 
to recognize tree species based on the leaf images, and exploring different aspects of the network

Member in the team: Chin Shiang Jin

Student ID: 002798503

1. Links/URLs to any videos you created and want to submit as part of your report.
N/A presentation was done in class

Link to dataset used 

https://www.kaggle.com/datasets/xhlulu/leafsnap-dataset/code

Link to trained model 

https://drive.google.com/drive/folders/1R1P_CFIVwD2S6DWgtSPMImn33Lax6wco?usp=sharing

2. What operating system and IDE you used to run and compile your code.

Operating system - Window

IDE - Visual Studio Code

3. Instructions for running your executables.
Set Up OS / IDE
- Download and install the Python Kernel
- Use terminal to install require libraries (matplotlib, torch, torchvision) using PIP command
- Refer https://pytorch.org/get-started/locally/ for further details
- There are 3 py files in total. Places all py file in the same folder, note some .py file will import class / function developed in other .py file
- For visual studio code, you can add the args in launch json file


4. Instructions for using the program. 

The main program is the model_training_testing.py, this allow running custom mode and model depending on the arguments set in commandline

The commandline arguments required are

model_training_testing.py -save <directory>

Available options are as follow

-save <directory> 	full absolute path of intended save directory for data (training or testing) (required)

-dir_model <directory> 	full absolute path of intended save directory for model and labels_dict(training or testing) (required)

-nepochs <number> 	string representation of number of epochs to run for the training model (integer, optional), default is 100. 

-train_size <size>	string representation of batch size for the training set (integer, optional), default is 200. 

-test_size <size> 	string representation of test size for the training set (integer, optional), default is 1000

-lrate 	<rate>		string representation of learning rate for the model during training (float, optional), default is 0.01. 

-momentum <rate>	string representation of the momentum for the model during training (float, optional), default is 0.5. 

-log_int <interval>	string representation of the log interval for the model during training (integer, optional), default is 100. 

-test_int <interval>string representation of the test interval for the model during training (integer, optional), default is 10

-target_size <number> string representation of the target size of the input images for the model (integer, optional), default is 44. All loaded images will be resized to this size. 
Note during testing mode, the target_size need to match the target_size used during training. 

-model <model>		string representation of selected model, currently choice of "2CNN", "3CNN", "Resnet50", default is "2CNN"

--grayscale 		set the grayscale option to True, to set it to False use --no-grayscale 

--training 		    set to training mode , to set it to testing mode use --no-training. 


5. Instructions for testing extensions 

The main program can be run in training mode and testing mode. This can be 

During training mode, the map of the target index to label will be loaded into a dictionary object and save as a json.txt file together with the model and optimizer states. 

During testing mode, the program will load the saved model state and optimizer state pretrained, and use the dictionary loaded from the json.txt file to match the target

Currently no production mode is supported. 


6. Time travel days used. 
N/A