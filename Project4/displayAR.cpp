/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 11/3/2023
  Description   : Program used to project the augmented reality object onto detected chessboard pattern
*/
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
    // Getting the variables from command line by using option prefix
    VideoCapture *capdev;
    int cameraIdx;
    try
    {
        cameraIdx = std::stoi(getOptionParam(argc, argv, "-cameraIdx="));
    }
    catch (std::exception)
    {
        // Set it to 0
        printf("Unable to read the cameraIdx, default will be used\n");
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

    // get some properties of the video frame
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);
    // Set default fps for saving
    int fps = (int)capdev->get(cv::CAP_PROP_FPS);
    printf("Captured fps: %d \n", fps);
    int savingFPS = fps > 0 ? std::min<int>(fps, 10) : 10;

    // Set default save path
    String defaultSavePath;
    try
    {
        defaultSavePath = getOptionParam(argc, argv, "-saveDir=");
        if (defaultSavePath.length() == 0)
        {
            throw std::invalid_argument("");
        }
    }
    catch (std::exception)
    {
        std::cout << "Could not read the default save path " << std::endl;
        exit(EXIT_FAILURE);
    }
    cout << "save path is " << defaultSavePath << endl;

    // get the selected mode
    int mode;
    try
    {
        mode = std::stoi(getOptionParam(argc, argv, "-mode="));
    }
    catch (std::exception)
    {
        // Set it to 0
        printf("Unable to read the selected mode, default will be used\n");
        mode = 0;
    }

    // Get the rows and cols of the chessboard pattern
    int nrow = 6;
    int ncol = 9;
    try
    {
        nrow = std::stoi(getOptionParam(argc, argv, "-rows="));
        ncol = std::stoi(getOptionParam(argc, argv, "-cols="));
    }
    catch (std::exception)
    {
        // Set it to 0
        printf("Unable to read the rows and cols, default will be used\n");
        nrow = 6;
        ncol = 9;
    }

    // Get the obj and mtl file for mode 1
    std::string objFilePath;
    std::string mtlFilePath;
    if (mode == 1)
    {
        try
        {
            objFilePath = getOptionParam(argc, argv, "-objFile=");
            mtlFilePath = getOptionParam(argc, argv, "-mtlFile=");
        }
        catch (std::exception)
        {
            std::cout << "Could not read the objFilePath or mtlFilePath " << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Get the target image for mode 2
    std::string warpImgPath;
    if (mode == 2)
    {
        try
        {
            warpImgPath = getOptionParam(argc, argv, "-image=");
        }
        catch (std::exception)
        {
            std::cout << "Could not read the warp image path " << std::endl;
            exit(EXIT_FAILURE);
        }
    }

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
    if (readIntrinsic(cameraMatrix, distCoeffs, defaultSavePath) != 0)
    {
        std::cout << "Unable to read camera instrinsic " << std::endl;
        exit(EXIT_FAILURE);
    };

    cv::Mat rvec;
    cv::Mat tvec;
    std::vector<cv::Vec3f> point_set;
    std::vector<Point2f> corner_set;

    // Initialize the point_set
    for (int y = 0; y < nrow; y++)
    {
        for (int x = 0; x < ncol; x++)
        {
            cv::Vec3f tempPt = Vec3f(x * 1.0F, y * 1.0F, 0);
            point_set.push_back(tempPt);
        }
    }

    // Initialize the axes points for mode 0
    std::vector<cv::Vec3f> axesPoints;
    axesPoints.push_back(Vec3f(ncol / 2.0F, nrow / 2.0F, 0));  // Center
    axesPoints.push_back(Vec3f(ncol * 1.0F, nrow / 2.0F, 0));  // x-edge
    axesPoints.push_back(Vec3f(ncol / 2.0F, nrow * 1.0F, 0));  // y-edge
    axesPoints.push_back(Vec3f(ncol / 2.0F, nrow / 2.0F, -5)); // z-edge

    // Initialize the obj object for mode 1, and the scale factor (default 100)
    OBJStruct object(cv::Scalar(255, 255, 0));
    int scaleAR = 100;
    // Initialize the rotation angle, yaw (alpha around z axis), pitch(beta around y axis), roll (gamma aroundx axis)
    int yaw = 0;
    int pitch = 0;
    int roll = 0;
    bool shownTrackBar = false;
    if (mode == 1)
    {
        cout << "mtlfile:" << mtlFilePath << endl;
        cout << objFilePath << endl;
        if (mtlFilePath.length() != 0)
        {
            object.loadFile(objFilePath, true, mtlFilePath);
        }
        else
        {
            cout << "loading file without mtl" << endl;
            object.loadFile(objFilePath);
        }
    }

    // initialize the corner points for mode 2
    // we need top-left, top-right, bottom-right, and bottom-left in the order
    // note the world coordinate used is based on the internal corner (0, 0, 0) for the first corner on top left
    // so we need to add the distance to get the outer corner
    std::vector<cv::Vec3f> cornerPoints;
    cornerPoints.push_back(Vec3f(-1, -1, 0));                   // top-left
    cornerPoints.push_back(Vec3f(ncol * 1.0F, -1, 0));          // top-right
    cornerPoints.push_back(Vec3f(ncol * 1.0F, nrow * 1.0F, 0)); // bottom-right
    cornerPoints.push_back(Vec3f(-1, nrow * 1.0F, 0));          // bottom-left

    // Retrieve the warp image
    Mat warpImg;
    std::vector<Point2f> warpCorners;
    if (mode == 2)
    {
        try
        {
            warpImg = cv::imread(warpImgPath, IMREAD_COLOR);
            warpCorners.push_back(Point2f(0, 0));
            warpCorners.push_back(Point2f(warpImg.cols - 1, 0));
            warpCorners.push_back(Point2f(warpImg.cols - 1, warpImg.rows - 1));
            warpCorners.push_back(Point2f(0, warpImg.rows - 1));
        }
        catch (std::exception)
        {
            std::cout << "Could not retrieve the warp image " << std::endl;
            exit(EXIT_FAILURE);
        }
    }

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
            cornerSubPix(gray, corner_set, Size(11, 11), Size(-1, -1),
                         TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 30, 0.1));
            solvePnP(point_set, corner_set, cameraMatrix, distCoeffs, rvec, tvec);
            if (mode == 0)
            {
                // demo mode, project the axes
                cout << "rvec: " << endl
                     << rvec << endl;
                cout << "tvec: " << endl
                     << tvec << endl;
                drawChessboardCorners(displayFrame, patternsize, Mat(corner_set), patternfound);
                std::vector<Point2f> arPoints;
                projectPoints(axesPoints, rvec, tvec, cameraMatrix, distCoeffs, arPoints);
                cv::arrowedLine(displayFrame, arPoints[0], arPoints[1], Scalar(255, 0, 0), 5);
                cv::arrowedLine(displayFrame, arPoints[0], arPoints[2], Scalar(0, 255, 0), 5);
                cv::arrowedLine(displayFrame, arPoints[0], arPoints[3], Scalar(0, 0, 255), 5);
            }
            else if (mode == 1)
            {
                // Virtual Object mode
                // Update the scaleAR factor
                scaleAR = std::max<int>(10, scaleAR);
                float scale = scaleAR / 100.0;
                object.projectObject(displayFrame, cameraMatrix, distCoeffs, rvec, tvec, scale, ncol / 2.0F, nrow / 2.0F, yaw, pitch, roll);
            }
            else if (mode == 2)
            {
                // Image overlay mode
                // Get the four corner of the chessboard from world coordinate to img coordinate
                std::vector<Point2f> cornerInImgs;
                projectPoints(cornerPoints, rvec, tvec, cameraMatrix, distCoeffs, cornerInImgs);
                // Get the homography matrix
                Mat homo = cv::findHomography(warpCorners, cornerInImgs);
                Mat warpedImg;
                // Transform the warp image, and overlay it onto the display frame
                cv::warpPerspective(warpImg, warpedImg, homo, displayFrame.size());
                overlayImg(warpedImg, displayFrame, cornerInImgs);
            }
        }

        // check if video saving is on
        if (videoSaving)
        {
            videoWriter.write(displayFrame);
        }

        cv::imshow("Video", displayFrame);
        // Show the trackbar for mode 1
        if (mode == 1 && !shownTrackBar)
        {
            cv::createTrackbar("Scale %", "Video", &scaleAR, 200);
            cv::createTrackbar("Zangle deg", "Video", &yaw, 360);
            cv::createTrackbar("Yangle deg", "Video", &pitch, 360);
            cv::createTrackbar("Xangle deg", "Video", &roll, 360);
            shownTrackBar = true;
        }
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
        // keystroke s save the current display frame
        case 's':
        {
            std::time_t timeStamp = std::time(nullptr);
            String finalPath = defaultSavePath + "screenshot_" + to_string(timeStamp) + ".png";
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

    return (0);
}