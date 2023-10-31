#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "utils.h"
#include "objLoader.h"

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

    cv::Mat capturedFrame;
    cv::Mat tempFrame;
    cv::Mat displayFrame;

    // Initialize all boolean flags for display option
    bool demo = false;
    bool videoSaving = false;

    cv::VideoWriter videoWriter;
    // Read the camera matrix and dist coeffs from file
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    readIntrinsic(cameraMatrix, distCoeffs, defaultSavePath);
    cv::Mat rvec;
    cv::Mat tvec;
    std::vector<cv::Vec3f> point_set;
    std::vector<Point2f> corner_set; // this will be filled by the detected corners

    for (int y = 0; y < 6; y++)
    {
        for (int x = 0; x < 9; x++)
        {
            cv::Vec3f tempPt = Vec3f(x, y, 0);
            point_set.push_back(tempPt);
        }
    }

    std::vector<cv::Vec3f> objectPoints;
    objectPoints.push_back(Vec3f(4.5, 2.5, 0));  // Center
    objectPoints.push_back(Vec3f(9, 2.5, 0));    // x-edge
    objectPoints.push_back(Vec3f(4.5, 6, 0));    // y-edge
    objectPoints.push_back(Vec3f(4.5, 2.5, -5)); // z-edge
    std::vector<Point2f> arPoints;
    OBJStruct object;
    object.loadFile("C:/CS5330_Assets/Project4/Jeep.obj");

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
            solvePnP(point_set, corner_set, cameraMatrix, distCoeffs, rvec, tvec);
            if (demo)
            {
                cout << "rvec: " << endl
                     << rvec << endl;
                cout << "tvec: " << endl
                     << tvec << endl;
                drawChessboardCorners(displayFrame, patternsize, Mat(corner_set), patternfound);
            }
            projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, arPoints);
            object.projectObject(displayFrame, cameraMatrix, distCoeffs, rvec, tvec);
            cv::arrowedLine(displayFrame, arPoints[0], arPoints[1], Scalar(255, 0, 0), 5);
            cv::arrowedLine(displayFrame, arPoints[0], arPoints[2], Scalar(0, 255, 0), 5);
            cv::arrowedLine(displayFrame, arPoints[0], arPoints[3], Scalar(0, 0, 255), 5);
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