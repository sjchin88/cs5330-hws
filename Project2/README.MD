CS5530 Fall 2023 (Seattle) Project 2 Readme
Name: Chin Shiang Jin
Student ID: 002798503

1. Links/URLs to any videos you created and want to submit as part of your report.
N/A

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
Run the buildDb program first to compute all feature vectors. The commandline arguments required are  
./buildDB.exe Argv[1] Argv[2] Argv[3] Argv[4]
where
Argv[1] 	<absolute path to the image database>
Argv[2]		<absolute path of the directory to store the csv file>
Argv[3]		<chosen method idx to compute features for an image F>
Argv[4]		<zoom factor to use>	

Available method to compute the features are as follow: 
1	Baseline Matching using 9x9 square in the middle of the image
2	Histogram Matching using normalized rg chromaticity histogram of 16 bins each
3	Multi-histogram Matching by separating the image into left and right half, and use a normalized rg chromaticity histogram of 16 bins each for each half
4 	Texture and Color matching, by using a whole image color histogram (of 8 bins R, G, B) and the histogram of gradient magnitude image. 
5	Texture and Color matching on zoomed image based on method 4. Need to supply a zoom factor between 0.1 to 1.0. 
6 	Texture and Color matching, by using a whole image color histogram (of 8 bins R, G, B) and the histogram of gabor filtered image. 
7	Texture and Color matching on zoomed image based on method 6. Need to supply a zoom factor between 0.1 to 1.0. 

To search using the target Image, run the searchImg program with the following commandline arguments
./searchImg.exe	Argv[1] Argv[2] Argv[3] Argv[4] Argv[5] Argv[6]
Argv[1] 	<absolute path to the target img>
Argv[2]		<absolute path of the directory to store the csv file>
Argv[3]		<chosen method idx to compute features for an image F>
Argv[4]		<chosen distance metric idx>
Argv[5] 	<N number of most similar images to be display>
Argv[6]		<zoom factor to use>

Available distance metric to compare the features are as follow: 
1	Sum-of-square method
2	histogram intersection

5. Instructions for testing extensions 
The extension implemented is the gabor filter bank. This can be tested by using the imgDisplay.exe from project 1. 
To recap, For imgDisplay program, the commandline are: 
Program name	 Argv[1]				   Argv[2]				 Argv[3]
./imgDisplay.exe <absolute path to the image to be loaded> <location to save the modified image> "custom caption text"

When the image or image frame is loaded,you can press keystroke 'a' to see the Gabor filtered image. 

Other Keystroke and extension effects are as follow:
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
t		Turn on/off the captioning, display the caption text on the image 
		If the captioning is already on, it will be turned off. If it is off, it will be turned on


5. Time travel days used. 
0 