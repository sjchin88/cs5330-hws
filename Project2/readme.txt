CS5530 Fall 2023 (Seattle) Project 2 Readme
Name: Chin Shiang Jin
Student ID: 002798503

1. Links/URLs to any videos you created and want to submit as part of your report.

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


4. Instructions for using the program. 
Run the buildDb program first to compute all feature vectors. The commandline arguments required are  
Argv[1] 	<absolute path to the image database>
Argv[2]		<absolute path of the directory to store the csv file>
Argv[3]		<chosen method idx to compute features for an image F>

Available method to compute the features are as follow: 
1	Baseline Matching using 9x9 square in the middle of the image
2	Histogram Matching using normalized rg chromaticity histogram of 16 bins each
3	Multi-histogram Matching by separating the image into left and rigth half, and use a normalized rg chromaticity histogram of 16 bins each for each half
4 	Texture and Color matching, by using a whole image color histogram (of 8 bins R, G, B) and the histogram of gradient magnitude image. 
5	  

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