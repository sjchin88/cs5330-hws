CS5530 Fall 2023 (Seattle) Project 4 Readme

Name: Chin Shiang Jin

Student ID: 002798503

1. Links/URLs to any videos you created and want to submit as part of your report.

https://drive.google.com/file/d/1MQW9u9EgTaN1BY9NRgZIt7AvK-CF4F5C/view?usp=sharing

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
	- https://github.com/Microsoft/vscode-cpptools
	- https://github.com/danielpinto8zz6/c-cpp-compile-run
	- https://github.com/twxs/vs.language.cmake
	- https://github.com/microsoft/vscode-cmake-tools
- Create a new project folder directory
- Inside the project folder directory, create new CMakeLists.txt file. Copy the content of attached CMakeLists file over. 
And set the OpenCV_DIR to the directory where you install the opencv. In my case it is set(OpenCV_DIR "C:/opencv/build")
- Place the files into the project directory. Note if you are using MVSC (Microsoft Visual Studio Compiler). 
You need to manually download the dirent.h file from https://github.com/tronkko/dirent
and placed it in your visual studio directory. 

- Build the project and selected cpp file. 
- Open a command prompt terminal, run the program with the required arguments in the build folder(or debug folder)


4. Instructions for using the program. 
Run the calibrateCamera program first, take at least 5 screenshot of images containing the detected chessboard pattern. 

Run the calibration to get the instrinsic parameters (camera matrix and distortion coefficients) and save them. 

The commandline arguments required are  

./buildDB.exe Argv[1] Argv[2] Argv[3] Argv[4] 

where

Argv[1] 	<camera idx of video feed>

Argv[2]		<absolute path of the directory to store the camera intrinsic parameters file>

Argv[3] 	<number of row for internal corner of the target chessboard pattern>

Argv[4] 	<number of col for internal corner of the target chessboard pattern>

Argv[3] & argv[4] are optional, the program can run without them. 

The available key stroke and extension effect for the calibrate camera program are as follow

Keystroke 	Effect

s			save the last calibration image

c			run the calibration and save the results

v			Turn on/off the video saving

q			quit the program


The next program to display augmented reality effect is the displayAR program. The commandline arguments required are

./displayAR.exe	-option1=xxx -option2=yyy -moreoption=zzz

Available options are as follow

-cameraIdx 	number of cameraIdx (optional)

-saveDir 	absolute path of the directory to store the camera intrinsic parameters file (required)

-rows		number of row for internal corner of the target chessboard pattern (optional)

-cols		number of col for internal corner of the target chessboard pattern (optional)

-mode		mode for the display (required), 
			
			0 = demo mode (shows only axes, print out the rotation and translation data in real time)

			1 = obj mode (will shows the object based on the obj file and or mtl file submitted)
			
			2 = warp image mode (will replace the chessboard pattern with chosen image)
			
-objFile	supply the .obj file for mode 1 (obj mode) (required)

-mtlFile	supply the .mtl file for mode 1 (obj mode) (optional)

-image		supply the image file for mode 2 (warp image mode)

The available key stroke and extension effect for the calibrate camera program are as follow

Keystroke 	Effect

s			save the last calibration image

v			Turn on/off the video saving

q			quit the program


The last program is for Harris Corner detection, The required commandline arguments are

./harrisCorner.exe	Argv[1] Argv[2] 

Where 

Argv[1] 	<camera idx of video feed>

Argv[2]		<absolute path of the directory for saving screenshot / video>

The available key stroke and extension effect for the video program are as follow

Keystroke 	Effect

s			save the screenshot of video

v			Turn on/off the video saving

q			quit the program

5. Instructions for testing extensions 

For the extension 1, project 3-D object, select mode 1 and supply the .obj file and .mtl file (see step 4 for the command option)

There will be 4 trackbars, which allow the user to scale the 3-D object, rotate the object around z-axis, y-axis and x-axis respectively. 
See the video link in 1 for the demonstration of the effect. 

For the extension 2, overlay another image onto the detected pattern, select mode 2 and supply the target image file. (see step 4 for the command option)

6. Time travel days used. 
0 