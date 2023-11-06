/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 11/3/2023
  Description   : Program used to perform camera calibration by
                - taking screenshot of images containing chessboard pattern
                - run the calibration to get the camera intrinsic parameters
                - save the camera instrinsic parameters into xml file
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

    // Set the number of row and column used for calibration
    int nrow = 6;
    int ncol = 9;
    if (argc >= 4)
    {
        // Get the row from argv[3], col from argv[4]
        try
        {
            nrow = std::stoi(argv[3]);
            ncol = std::stoi(argv[4]);
        }
        catch (std::exception)
        {
            std::cout << "Could not read the number of row and col, default will be used" << std::endl;
            nrow = 6;
            ncol = 9;
        }
    }

    // Initialize the variable required
    Mat capturedFrame;
    Mat tempFrame;
    Mat displayFrame;
    char selection = 'R';

    // Variables for calibration,
    // point_set refer to object coordinate in world coordinate system based on chessboard
    // corner_set is the detected chessboard corners
    // lastCalibrationImg holds the last image frame which detects the chessboard
    // point_list & corner_list hold the list of found point_set and corner_set used for calibration
    // image_list store the name lists of the images used for calibration
    Mat lastCalibrationImg;
    vector<cv::Vec3f> point_set;
    vector<vector<cv::Vec3f>> point_list;
    vector<Point2f> corner_set;
    vector<vector<cv::Point2f>> corner_list;
    vector<string> image_list;

    // Default chessboard used is 6 rows x 9 columns
    for (int y = 0; y < nrow; y++)
    {
        for (int x = 0; x < ncol; x++)
        {
            cv::Vec3f tempPt = Vec3f(x, y, 0);
            point_set.push_back(tempPt);
        }
    }

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

        // Set interior number of corners, columns, rows
        Size patternsize(ncol, nrow);

        // Try to find the pattern, from OpenCV documentation
        // CALIB_CB_FAST_CHECK saves a lot of time on images
        // that do not contain any chessboard corners
        bool patternfound = findChessboardCorners(gray, patternsize, corner_set,
                                                  CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
        if (patternfound)
        {
            // refine the corner
            cornerSubPix(gray, corner_set, Size(11, 11), Size(-1, -1),
                         TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 30, 0.1));
            // draw the found chessboard
            drawChessboardCorners(displayFrame, patternsize, Mat(corner_set), patternfound);
            // save the last drawn frame in a temp frame for use later
            lastCalibrationImg = displayFrame;
            // Debug, print number of corners found and first corner coordinate
            std::cout << "Number of corner found: " << corner_set.size() << std::endl;
            std::cout << "Coordinates of first corner: " << corner_set[0] << std::endl;
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
            if (point_set.size() != 0 && corner_set.size() != 0)
            {
                vector<cv::Vec3f> point_set_temp(point_set);
                point_list.push_back(point_set_temp);
                vector<Point2f> corner_set_temp(corner_set);
                corner_list.push_back(corner_set_temp);
                corner_set.clear();
                std::time_t timeStamp = std::time(nullptr);
                String finalPath = defaultSavePath + "screenshot_" + to_string(point_list.size()) + "_" + to_string(timeStamp);
                image_list.push_back(finalPath);
                finalPath += ".png";
                imwrite(finalPath, lastCalibrationImg);
            }
            break;
        }

        // keystroke c to run the calibration
        case 'c':
        {
            calibrateNSave(lastCalibrationImg, point_list, corner_list, image_list, defaultSavePath);
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