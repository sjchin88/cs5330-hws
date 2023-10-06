#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cvdef.h>
#include <vector>
#include <ctime>
#include <iostream>
#include "filters.h"

using namespace std;
using namespace cv;

/**
 * Main function
 */
int main(int argc, char *argv[])
{
    // Getting the variables from command line
    // argv[1] = image_path, argv[2] = defaultSavePath, argv[3] = captionText to use
    string image_path = samples::findFile(argv[1]);
    // Debug Code string image_path = samples::findFile("C:/CS5330_Assets/lenna.png");
    cout << "image path is " << image_path << endl;

    Mat img = imread(image_path, IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    String defaultSavePath;
    try
    {
        defaultSavePath = argv[2];
    }
    catch (std::exception)
    {
        std::cout << "Could not read the default save path " << std::endl;
        return 1;
    }
    cout << "save path is " << defaultSavePath << endl;

    String captionText;
    try
    {
        captionText = argv[3];
    }
    catch (std::exception)
    {
        // set it to default text
        captionText = "Hahaha";
    }
    cout << "caption text is " << captionText << endl;

    // Initialize variables required
    Mat displayImg;
    Mat tempImg;
    char selection = 'R';
    bool captioning = false;

    // Run the while loop
    while (true)
    {
        displayImg = img;

        // Convert img based on current selections
        switch (selection)
        {
        case 'G':
            // Task 3: Use the cvtColor() function to grayscale the image
            cvtColor(img, displayImg, COLOR_BGR2GRAY);
            break;

        case 'H':
            if (greyscale(img, tempImg) == 0)
            {
                displayImg = tempImg;
            }
            break;

        case 'B':
            if (blur5x5(img, tempImg) == 0)
            {
                displayImg = tempImg;
            }
            break;

        case 'X':
            if (sobelX3x3(img, tempImg) == 0)
            {
                // Output of sobelX3x3 filter are stored in 16S, convert it for display
                Mat tempImg2;
                convertScaleAbs(tempImg, tempImg2);
                displayImg = tempImg2;
            }
            break;

        case 'Y':
            if (sobelY3x3(img, tempImg) == 0)
            {
                // Output of sobelY3x3 filter are stored in 16S, convert it for display
                Mat tempImg2;
                convertScaleAbs(tempImg, tempImg2);
                displayImg = tempImg2;
            }
            break;

        case 'M':
        {
            Mat tempImgX;
            Mat tempImgY;
            Mat tempImgM;
            if (sobelX3x3(img, tempImgX) == 0 && sobelY3x3(img, tempImgY) == 0)
            {
                if (magnitude(tempImgX, tempImgY, tempImgM) == 0)
                {
                    // Output of magnitude filter are stored in 16S, convert it for display
                    Mat tempImg2;
                    convertScaleAbs(tempImgM, tempImg2);
                    displayImg = tempImg2;
                }
            }
            break;
        }

        case 'I':
        {
            int defaultLevel = 15;
            if (blurQuantize(img, tempImg, defaultLevel) == 0)
            {
                displayImg = tempImg;
            }
            break;
        }

        case 'C':
        {
            int defaultLevel = 15;
            int defaultMagThreshold = 15;
            if (cartoon(img, tempImg, defaultLevel, defaultMagThreshold) == 0)
            {
                displayImg = tempImg;
            }
            break;
        }

        case 'N':
            if (negative(img, tempImg) == 0)
            {
                displayImg = tempImg;
            }
            break;

        case 'A':
        {
            cv::Size ksize = cv::Size(31, 31);
            vector<float> sigmas;
            sigmas.push_back(4.0);
            vector<float> thetas;
            for (float i = 0.0; i < CV_PI; i += CV_PI / 16)
            {
                thetas.push_back(i);
            }
            vector<float> lambdas;
            lambdas.push_back(10.0);
            vector<float> gammas;
            gammas.push_back(0.5);
            if (gaborFiltering(img, tempImg, ksize, sigmas, thetas, lambdas, gammas) == 0)
            {
                displayImg = tempImg;
            }
            break;
        }

        default:
            break;
        }

        // check if captioning is on
        if (captioning)
        {
            cv::Point2i btmLeft(50, 100);
            cv::putText(displayImg, captionText, btmLeft, cv::FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0));
        }

        // After applied all effect, show the image
        imshow("Display window", displayImg);

        // see if there is a waiting keystroke
        char key = cv::waitKey(0);

        switch (key)
        {

        // Task 3: keystroke g turn the current frame into greyscale
        case 'g':
            selection = 'G';
            break;

        // Task 4: keystroke h turn the current frame into custom greyscale
        case 'h':
            selection = 'H';
            break;

        // Task 5: keystroke h turn the current frame into blur5x5
        case 'b':
            selection = 'B';
            break;

        // Task 6: keystroke x, y turn the current frame into Sobel Filter
        case 'x':
            selection = 'X';
            break;

        case 'y':
            selection = 'Y';
            break;

        // Task 7: keystroke m turn the current frame into gradient magnitude image
        case 'm':
            selection = 'M';
            break;

        // Task 8: keystroke i turn the current frame into blurs and quantizes image
        case 'i':
            selection = 'I';
            break;

        // Task 9: keystroke c turn the current frame into cartoon image
        case 'c':
            selection = 'C';
            break;

        // Task 10: keystroke n turn the current frame into negative of itself
        case 'n':
            selection = 'N';
            break;

        // Extension 1: keystroke t to turn on/off the captioning of the image
        case 't':
            captioning = !captioning;
            break;

        // Project2 test: keystroke a to turn current frame into gabor filtered image
        case 'a':
            selection = 'A';
            break;

        // keystroke r to change the selection back to normal
        case 'r':
            selection = 'R';
            break;

        default:
            break;
        }

        if (key == 's')
        {
            std::time_t timeStamp = std::time(nullptr);
            String finalPath = defaultSavePath + "saveImg_" + to_string(timeStamp) + ".png";
            imwrite(finalPath, displayImg);
        }
        // The program will quit if the user press 'q'
        if (key == 'q')
        {
            return 0;
        }
    }

    return 0;
}