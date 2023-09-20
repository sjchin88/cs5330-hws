#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <ctime>
#include "filters.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    VideoCapture *capdev;

    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        return (-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    namedWindow("Video", 1); // identifies a window
    Mat capturedFrame;
    Mat tempFrame;
    Mat displayFrame;

    bool greying = false;
    while (true)
    {
        *capdev >> capturedFrame; // get a new frame from the camera, treat as a stream

        if (capturedFrame.empty())
        {
            printf("frame is empty\n");
            break;
        }

        if (greying)
        {
            // CSJ: Use the cvtColor() function to grayscale the image
            // cvtColor(capturedFrame, displayFrame, COLOR_BGR2GRAY);
            Mat tempFrame(capturedFrame.size(), CV_8UC1, Scalar(0));
            if (greyscale(capturedFrame, tempFrame) == 0)
            {
                displayFrame = tempFrame;
            }
            else
            {
                displayFrame = capturedFrame;
            }
        }
        else
        {
            displayFrame = capturedFrame;
        }

        cv::imshow("Video", displayFrame);
        // see if there is a waiting keystroke
        char key = cv::waitKey(10);

        // Task 2: keystroke s save the current display frame
        if (key == 's')
        {
            std::time_t timeStamp = std::time(nullptr);
            String path = "C:/CS5330_Assets/screenshot";
            String finalPath = path + to_string(timeStamp) + ".png";
            imwrite(finalPath, displayFrame);
        }

        // Task 3: keystroke g turn the current frame into greyscale
        // type r to change back
        if (key == 'g')
        {
            greying = true;
        }

        if (key == 'r')
        {
            greying = false;
        }

        // keystroke q quit the program
        if (key == 'q')
        {
            printTest(2);
            break;
        }
    }

    delete capdev;
    return (0);
}