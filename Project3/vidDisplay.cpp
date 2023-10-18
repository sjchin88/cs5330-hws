/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/06/2023
  Description   : Filters from project 1.
                * Filters used in project 2:
                * sobelX, sobelY and magnitude filter
                * New filter added : Gabor filter
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "filters.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    VideoCapture *capdev;

    // Set the cameraIdx from the commandline argument
    int cameraIdx;
    try
    {
        cameraIdx = std::stoi(argv[1]);
    }
    catch (std::exception)
    {
        // Set it to 0
        cameraIdx = 0;
    }
    cout << "camera idx is " << cameraIdx << endl;

    // open the video device
    capdev = new cv::VideoCapture(cameraIdx);
    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        return (-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);
    int fps = (int)capdev->get(cv::CAP_PROP_FPS);
    printf("Captured fps: %d \n", fps);

    // Set default fps for saving
    int savingFPS = 30;
    if (fps > 0)
    {
        savingFPS = fps;
    }

    // Set default save path
    String defaultSavePath;
    try
    {
        // defaultSavePath = argv[2];
    }
    catch (std::exception)
    {
        std::cout << "Could not read the default save path " << std::endl;
        return 1;
    }
    cout << "save path is " << defaultSavePath << endl;

    // Initialize variables required
    namedWindow("Video", 1); // identifies a window
    Mat capturedFrame;
    Mat tempFrame;
    Mat displayFrame;
    char selection = 'R';
    bool videoSaving = false;
    cv::VideoWriter videoWriter;

        // Run the while loop
    while (true)
    {
        *capdev >> capturedFrame; // get a new frame from the camera, treat as a stream

        if (capturedFrame.empty())
        {
            printf("frame is empty\n");
            break;
        }

        // Default is to set the displayFrame as capturedFrame
        displayFrame = capturedFrame;

        // Try to apply the selected effect to the capturedFrame
        // and set the displayFrame to modified frame only if successful
        switch (selection)
        {

        default:
            displayFrame = capturedFrame;
            break;
        }

        // check if video saving is on
        if (videoSaving)
        {
            videoWriter.write(displayFrame);
        }

        cv::imshow("Video", displayFrame);

        // see if there is a waiting keystroke every 33ms
        char key = cv::waitKey(33);

        switch (key)
        {

        //  keystroke v turn on saving video to the file
        case 'v':
        {
            // if videoSaving is on, turn it off
            if (videoSaving)
            {
                videoWriter.release();
                videoSaving = false;
            }
            else
            {
                // Set the video writer
                std::time_t timeStamp = std::time(nullptr);
                String videoPath = defaultSavePath + "video_" + to_string(timeStamp) + ".avi";
                /*  cv::VideoWriter::VideoWriter (const String &filename, int fourcc, double fps,Size frameSize,bool isColor = true
                        )	 */
                videoWriter = cv::VideoWriter(videoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), savingFPS, refS);
                videoSaving = true;
            }

            break;
        }

        // keystroke r to change the selection back to normal
        case 'r':
            selection = 'R';
            break;

        default:
            break;
        }

        // Task 2: keystroke s save the current display frame
        if (key == 's')
        {
            std::time_t timeStamp = std::time(nullptr);
            String finalPath = defaultSavePath + "screenshot_" + to_string(timeStamp) + ".png";
            imwrite(finalPath, displayFrame);
        }

        // keystroke q quit the program
        if (key == 'q')
        {
            break;
        }
    }

    delete capdev;
    return (0);
}