/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   : 2-D object recognition for real time video feed
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <iostream>
#include "utils.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    // check for sufficient arguments
    if (argc < 2)
    {
        printf("usage: %s <cameraIdx> <defaultSavePath>\n", argv[0]);
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
        // Set it to 0
        cameraIdx = 0;
    }
    cout << "camera idx is " << cameraIdx << endl;

    // open the video device
    capdev = new cv::VideoCapture(cameraIdx);
    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        exit(EXIT_FAILURE);
    }

    // get some properties of the image
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

    Mat capturedFrame;
    Mat tempFrame;
    Mat displayFrame;
    Mat lastCalibrationImg;
    char selection = 'R';
    vector<cv::Vec3f> point_set;
    vector<vector<cv::Vec3f>> point_list;
    vector<Point2f> corner_set; // this will be filled by the detected corners
    vector<vector<cv::Point2f>> corner_list;
    for (int y = 0; y < 6; y++)
    {
        for (int x = 0; x < 9; x++)
        {
            cv::Vec3f tempPt = Vec3f(x, y, 0);
            point_set.push_back(tempPt);
        }
    }
    // Initialize all boolean flags for display option
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
        Mat gray; // source image
        cv::cvtColor(capturedFrame, gray, cv::COLOR_BGR2GRAY);
        Size patternsize(9, 6); // interior number of corners, columns, rows

        // CALIB_CB_FAST_CHECK saves a lot of time on images
        // that do not contain any chessboard corners
        bool patternfound = findChessboardCorners(gray, patternsize, corner_set,
                                                  CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
        if (patternfound)
        {
            cornerSubPix(gray, corner_set, Size(11, 11), Size(-1, -1),
                         TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 30, 0.1));
            drawChessboardCorners(displayFrame, patternsize, Mat(corner_set), patternfound);
            lastCalibrationImg = displayFrame;
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

        default:
            break;
        }

        // keystroke s save the current display frame
        if (key == 's')
        {
            vector<cv::Vec3f> point_set_temp = point_set;
            point_list.push_back(point_set_temp);
            vector<Point2f> corner_set_temp = corner_set;
            corner_list.push_back(corner_set_temp);
            std::time_t timeStamp = std::time(nullptr);
            String finalPath = defaultSavePath + "screenshot_" + to_string(timeStamp) + ".png";
            imwrite(finalPath, lastCalibrationImg);
        }

        // keystroke c to run the calibration
        if (key == 'c')
        {
            calibrateNSave(lastCalibrationImg, point_list, corner_list, defaultSavePath);
        }

        // keystroke q quit the program
        if (key == 'q')
        {

            for (auto corner_set : point_list)
            {
                cout << corner_set.size() << endl;
                for (auto point : corner_set)
                {
                    cout << point << "; ";
                }
                cout << endl;
            }
            for (auto corner_set : corner_list)
            {
                cout << corner_set.size() << endl;
                for (auto point : corner_set)
                {
                    printf("x: %f, y: %f ;", point.x, point.y);
                }
                cout << endl;
            }
            calibrateNSave(lastCalibrationImg, point_list, corner_list, defaultSavePath);
            break;
        }
    }

    delete capdev;
    return (0);
}