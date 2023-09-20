#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
int main()
{
    // Change the file path of the image to where you place it
    std::string image_path = samples::findFile("C:/CS5330_Assets/lenna.png");
    Mat img = imread(image_path, IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    while (true)
    {
        imshow("Display window", img);
        // Wait for a keystroke in the window
        int k = waitKey(0);

        if (k == 's')
        {
            imwrite("C:/CS5330_Assets/screenshot.png", img);
        }
        // The program will quit if the user press 'q'
        if (k == 'q')
        {
            return 0;
        }
    }

    return 0;
}