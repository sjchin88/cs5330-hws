CS5530 Project 1
Name: Chin Shiang Jin

1. Links/URLs to any videos you created and want to submit as part of your report.

2. What operating system and IDE you used to run and compile your code.
Operating system - Window
IDE - Visual Studio Code

3. Instructions for running your executables.
Set Up OS / IDE
1. Download and install Microsoft Visual Studio 2022 Community Edition with C++ support (use for compiler)
2. Download OpenCV Window installer and install using it on directory of your choice
3. Add the OpenCV bin directory (example: C:\opencv\build\x64\vc16\bin) to the Path of your system environmental variable
4. Download CMake Window installer and install it on directory of your choice (CMake is used to manage dependency)
5. Make sure the CMake bin directory is added to the Path of your system environmental variable
6. Install required visual studio code extensions as below
https://github.com/Microsoft/vscode-cpptools
https://github.com/danielpinto8zz6/c-cpp-compile-run
https://github.com/twxs/vs.language.cmake
https://github.com/microsoft/vscode-cmake-tools
7. Create a new project folder directory
8. Inside the project folder directory, create new CMakeLists.txt file. Copy the content of attached CMakeLists file over. 
And set the OpenCV_DIR to the directory where you install the opencv. In my case it is set(OpenCV_DIR "C:/opencv/build")
9. Place the files into the project directory. 
10. Build and run. 

Instructions for testing any extensions you completed.
Whether you are using any time travel days, and how many.