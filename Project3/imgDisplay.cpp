/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   : imgDisplay program from Project 1
                    used to verify the effect of all implemented functions
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cvdef.h>
#include <vector>
#include <iostream>
#include "filters.h"
#include "utils.h"

using namespace std;
using namespace cv;

/**
 * Main function
 */
int main(int argc, char *argv[])
{
    // Getting the variables from command line
    // argv[1] = image_path, argv[2] = defaultSavePath
    // string image_path = samples::findFile(argv[1]);
    string image_path = samples::findFile("C:/CS5330_Assets/Proj03Examples/img1p3.png");
    cout << "image path is " << image_path << endl;

    Mat img = imread(image_path, IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    Mat dst;
    thresholdFilter(img, dst);
    imshow("threshold image", dst);
    Mat dst1;
    morphologyFilter(dst, dst1);
    imshow("morpho image", dst1);
    vector<cv::Mat> regionList;
    vector<float> features;
    getConnectedComponentRegions(dst1, regionList, true);
    for (int id = 0; id < regionList.size(); id++)
    {
        string title = "region id: " + id;
        imshow(title, regionList[id]);

        getOrientedBoundingBoxStat(regionList[id], features);
        Moments regionM = cv::moments(regionList[id], true);
        cout << "area using moments: " << regionM.m00 << endl;
    }

    String defaultSavePath = ("C:/CS5330_Assets/Proj03Examples/");
    ;
    /* try
    {
        defaultSavePath = argv[2];
    }
    catch (std::exception)
    {
        std::cout << "Could not read the default save path " << std::endl;
        return 1;
    }
    cout << "save path is " << defaultSavePath << endl; */

    // Initialize variables required
    Mat displayImg;
    Mat tempImg;
    char selection = 'R';

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

        case 'B':
            if (blur5x5(img, tempImg) == 0)
            {
                displayImg = tempImg;
            }
            break;

        default:
            break;
        }

        // After applied all effect, show the image
        imshow("Original Image", displayImg);

        // see if there is a waiting keystroke
        char key = cv::waitKey(0);

        switch (key)
        {

        // keystroke g turn the current frame into greyscale
        case 'g':
            selection = 'G';
            break;

        // keystroke b turn the current frame into blur5x5
        case 'b':
            selection = 'B';
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