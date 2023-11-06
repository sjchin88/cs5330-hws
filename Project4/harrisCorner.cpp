/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 11/3/2023
  Description   : Program for Task 7, detect the Harris corners
*/

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <time.h>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    // check for sufficient arguments
    if (argc < 2)
    {
        printf("usage: %s <cameraIdx> <defaultSavePath> <number of row> <number of column>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Getting the variables from command line
    // argv[1] = cameraIdx, argv[2] = defaultSavePath,
    // Set the cameraIdx from argv[1], and set the videocapture element
    VideoCapture *capdev;
    int cameraIdx;
    try
    {
        cameraIdx = std::stoi(argv[1]);
    }
    catch (std::exception)
    {
        // Set it to 0 for default
        cameraIdx = 0;
    }
    cout << "camera idx is " << cameraIdx << endl;

    // try open the video device
    capdev = new cv::VideoCapture(cameraIdx);
    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        exit(EXIT_FAILURE);
    }

    // get some properties of the video frame
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);
    int fps = (int)capdev->get(cv::CAP_PROP_FPS);
    printf("Captured fps: %d \n", fps);

    // Set default fps for saving
    int savingFPS = 10;
    if (fps > 0)
    {
        savingFPS = fps;
    }

    // Set default save path from argv[2]
    String defaultSavePath;
    try
    {
        defaultSavePath = argv[2];
    }
    catch (std::exception)
    {
        std::cout << "Could not read the default save path " << std::endl;
        exit(EXIT_FAILURE);
    }
    cout << "save path is " << defaultSavePath << endl;

    // Initialize the variable required
    Mat capturedFrame;
    Mat tempFrame;
    Mat displayFrame;
    char selection = 'R';

    // Initialize  boolean flag for video saving
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

        // Convert the source image into gray scale
        Mat gray;
        cv::cvtColor(capturedFrame, gray, cv::COLOR_BGR2GRAY);

        // Harris corner function take single channel image, so we need to convert it to single channel
        Mat singleGray;
        gray.convertTo(singleGray, CV_8UC1);
        // Try to find the harris corner
        Mat harrisFrame;
        cv::cornerHarris(singleGray, harrisFrame, 2, 3, 0.04);
        cv::dilate(harrisFrame, harrisFrame, NULL);
        // Get the max, to mark the harris corner
        double min, max;
        cv::minMaxIdx(harrisFrame, &min, &max);
        max = 0.01 * max;
        for (int row = 0; row < displayFrame.rows; row++)
        {
            for (int col = 0; col < displayFrame.cols; col++)
            {
                if (harrisFrame.at<float>(row, col) > max)
                {
                    displayFrame.at<cv::Vec3b>(row, col) = Vec3b(0, 255, 0);
                }
            }
        }
        // check if video saving is on
        if (videoSaving)
        {
            videoWriter.write(displayFrame);
        }

        // show the video
        cv::imshow("Video", displayFrame);

        // see if there is a waiting keystroke every 33ms
        char key = cv::waitKey(33);

        switch (key)
        {

        //  keystroke v turn on/off saving video to the file
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

        // keystroke s save the last found points, chessboard corners into the list and
        // save the last image containing chessboard pattern into the save location
        case 's':
        {
            std::time_t timeStamp = std::time(nullptr);
            String finalPath = defaultSavePath + "screenshot_" + to_string(timeStamp);
            finalPath += ".png";
            imwrite(finalPath, displayFrame);
            break;
        }

        // keystroke q quit the program
        case 'q':
        {
            delete capdev;
            exit(EXIT_SUCCESS);
        }

        default:
            break;
        }
    }

    delete capdev;
    return (0);
}