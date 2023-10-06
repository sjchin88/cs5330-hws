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
- Build the project and selected cpp file. 
- Open a command prompt terminal, run the program with the required arguments


4. Instructions for using the program. 
Run the buildDb program first in the build folder (either the debug or actual build folder where the buildDb.exe reside)
to compute all feature vectors. Example command line argument is follow
Argv[0]		
./buildDb.exe Argv[1] Argv[2] Argv[3] Argv[4]
The commandline arguments required are  
Argv[1] 	<absolute path to the image database>
Argv[2]		<absolute path of the directory to store the csv file>
Argv[3]		<chosen method idx to compute features for an image F>
Argv[4]		<zoom factor, required only if option 5 / 7 is selected in argv[3]

Available method to compute the features are as follow: 
1	Baseline Matching using 9x9 square in the middle of the image
2	Histogram Matching using normalized rg chromaticity histogram of 16 bins each
3	Multi-histogram Matching by separating the image into left and rigth half, and use a normalized rg chromaticity histogram of 16 bins each for each half
4 	Texture and Color matching, by using a whole image color histogram (of 8 bins R, G, B) and the histogram of gradient magnitude image. 
5	Texture and Color matching based on method 4 using only a crop portion of the image. Size of the crop portion determined by the zoom factor argument
6 	Texture and Color matching, by using a whole image color histogram (of 8 bins R, G, B) and the histogram of gabor filtered image. 
7	Texture and Color matching based on method 6 using only a crop portion of the image. Size of the crop portion determined by the zoom factor argument

For the searchImg program. Run it in the build folder similar to the buildDb program. Example command line argument is follow:
Argv[0]		
./searchImg.exe Argv[1] Argv[2] Argv[3] Argv[4] Argv[5]
The commandline arguments required are  
Argv[1] 	<absolute path to the target image>
Argv[2]		<absolute path of the directory to store the csv file>
Argv[3]		<chosen method idx to compute features for an image F>
Argv[4]		<Top N result to be displayed>
Argv[5]		<zoom factor, required only if option 5 / 7 is selected in argv[3]
The available method option in argv[3] is the same as the method option available for the buildDb program. 
Note the argv[5] zoom factor must be the same as the zoom factor used when computing the images database.

5. Instructions for testing extensions 
Extension features implemented: Compute of garbon filtered image.
Using the updated imgDisplay program from project 1, run it in the build folder with the following arguments
For imgDisplay program: 
Program name	 Argv[1]				   Argv[2]				 Argv[3]
./imgDisplay.exe <absolute path to the image to be loaded> <location to save the modified image> "custom caption text"

Once the image is shown, pressing keystroke 'a' will show the gabor filtered image. 

Other available key strokes for reference
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
N/A