CS5530 Fall 2023 (Seattle) Project 3 Readme
Name: Chin Shiang Jin
Student ID: 002798503

1. Links/URLs to any videos you created and want to submit as part of your report.
See https://drive.google.com/file/d/1LmaWQyen2sSZ1dRzkEpMwjq2YCAMGdsd/view?usp=sharing

2. What operating system and IDE you used to run and compile your code.
Operating system - Window
IDE - Visual Studio Code

3. Instructions for running your executables.
Set Up OS / IDE
- Download and install Microsoft Visual Studio 2022 Community Edition with C++ support (use for compiler)
- Download OpenCV Window installer and install using it on directory of your choice
- Add the OpenCV bin directory (example: C:\opencv\build\x64\vc16\bin) to the Path of your system environmental variable
- Download CMake Window installer and install it on directory of your choice (CMake is used to manage dependency)
- Make sure the CMake bin directory is added to the Path of your system environmental variable
- Install required visual studio code extensions as below
https://github.com/Microsoft/vscode-cpptools
https://github.com/danielpinto8zz6/c-cpp-compile-run
https://github.com/twxs/vs.language.cmake
https://github.com/microsoft/vscode-cmake-tools
- Create a new project folder directory
- Inside the project folder directory, create new CMakeLists.txt file. Copy the content of attached CMakeLists file over. 
And set the OpenCV_DIR to the directory where you install the opencv. In my case it is set(OpenCV_DIR "C:/opencv/build")
- Place the files into the project directory. 

Note if you are using MVSC (Microsoft Visual Studio Compiler). 
You need to manually download the dirent.h file from https://github.com/tronkko/dirent
and placed it in your visual studio directory. 

- Build the project and selected cpp file. 
- Open a command prompt terminal, run the program with the required arguments in the build folder(or debug folder)


4. Instructions for using the program. 
Run the buildDb program first with training sets provided to compute all feature vectors for known objects. The commandline arguments required are  
./buildDB.exe Argv[1] Argv[2] Argv[3] Argv[4]
where
Argv[1] 	<absolute path to the image directory>
Argv[2]		<absolute path of the directory to store the csv file>

Run the imgDisplay program for 2-D object recognition using stilled image. The required commandline arguments are
./imgDisplay.exe	Argv[1] Argv[2] Argv[3] Argv[4] 
Argv[1] 	<absolute path to the target img directory>
Argv[2]		<absolute path of the directory storing the object DB csv>
Argv[3] 	<top N number of connected region to be identified>
Argv[4] 	<k-value for K-Nearest Neighbor matching>
Argv[3] & argv[4] are optional, the program can run without them. 

The available key stroke and extension effect for the image program are as follow
Keystroke 	Effect
i			(Task 1) display the interim image after each preprocessing stage
t 			(Task 1) Turn on/off the thresholded image display
m 			(Task 2) Turn on/off the image after morphological operation
c 			(Task 3) Turn on/off the image colored by connected regions
z			(Task 4) Turn on/off the effect showing the axis of the least central moment and the oriented bounding box found
x 			(Task 4) Turn on/off the compute features text caption
n 			(Task 5) Prompt the user for a name/label input via the terminal, and then store the 
			feature vector for the current object along with its label into a file
a  			move to next image in the directory
q			quit the program

Run the imgDisplay program for 2-D object recognition using video feed. The required commandline arguments are
./vidDisplay.exe	Argv[1] Argv[2] Argv[3] Argv[4] Argv[5]
Argv[1] 	<camera idx of video feed>
Argv[2]		<absolute path of the directory for saving screenshot / video>
Argv[3]		<absolute path of the directory storing the object DB csv>
Argv[4] 	<top N number of connected region to be identified>
Argv[5] 	<k-value for K-Nearest Neighbor matching>
Argv[4] & argv[5] are optional, the program can run without them. 

The available key stroke and extension effect for the video program are as follow
Keystroke 	Effect
t 			(Task 1) Turn on/off the thresholded image display
c 			(Task 3) Turn on/off the image colored by connected regions
z			(Task 4) Turn on/off the effect showing the axis of the least central moment and the oriented bounding box found
x 			(Task 4) Turn on/off the compute features text caption
s			save the screenshot of video
v			Turn on/off the video saving
q			quit the program

5. Instructions for testing extensions 
To detect multiple objects at once, just capture multiple objects using the video camera
To detect unknown object, just place a random object not available in the existing DB for classification
To use the system to quickly update the DB, run the imgDisplay program, 
press keystroke n to enter a new label and save current object statistic to the Object DB
For other usage, refer to part 4





5. Time travel days used. 
0 