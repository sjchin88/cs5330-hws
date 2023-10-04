CS5530 Fall 2023 (Seattle) Project 1 Readme
Name: Chin Shiang Jin
Student ID: 002798503

1. Links/URLs to any videos you created and want to submit as part of your report.
Video for extension 2: https://drive.google.com/file/d/1GWSkABoqpoGv5oN0X3sKjnfyJc8T6Ry5/view?usp=sharing

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
- Build the project and selected cpp file. 
- Open a command prompt terminal, run the program with the required arguments
For imgDisplay program: 
Program name	 Argv[1]				   Argv[2]				 Argv[3]
./imgDisplay.exe <absolute path to the image to be loaded> <location to save the modified image> "custom caption text"

For vidDisplay program:
Program name	 Argv[1]	     Argv[2]			       Argv[3]
./vidDisplay.exe <int of camera idx> <location to save the screenshot> "custom caption text"

4. Instructions for using the program. 
When the image or image frame is loaded, you can pressed the following keys on your keyboard for the desired effect
Keystroke 	Effect
s		(Task 1 & 2) save the image with effect (for imgDisplay) or save the screenshot of video with effect (for vidDisplay)
q		(Task 1 & 2) quit the program
g		(Task 3) Turn the image into grayscale using standard openCV grayscale function
h		(Task 4) Turn the image into grayscale using custom grayscale function
b		(Task 5) Applied 5x5 Gaussian filter to the image
x		(Task 6) Applied sobelX effect to the image
y		(Task 6) Applied sobelY effect to the image
m		(Task 7) Applied gradient magnitude effect to the image
i		(Task 8) Applied blurs and quantizes effect to the image
c		(Task 9) Applied carton effect to the image
n		(Task 10) Turn the current image into the negative of itself
r		Change to no effect

5. Instructions for testing extensions 
Extension: Implement the effects for still images and enable the user to save the modified images.
All the keystroke listed in 4 can be used for the imgDisplay program as well. 

Other Keystroke and extension effects are as follow:
Keystroke 	Effect
t		Turn on/off the captioning, display the caption text on the image (work for both imgDisplay and vidDisplay program)
		If the captioning is already on, it will be turned off. If it is off, it will be turned on
v		Turn on/off the video saving, only work for vidDisplay program. 
		If the saving is already on, it will be turned off. If it is off, it will be turned on

5. Time travel days used. 
0 - With the assumption that the deadline is end of Friday (11.59pm pst)
1 - if past the actual deadline, please deduct 1 used. 