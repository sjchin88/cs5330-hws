/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 11/3/2023
  Description   : Utility functions required for Project 4
*/
#include "utils.h"

using namespace std;
using namespace cv;

/**
 * Function to calibrate the camera intrinsic parameters, get the camera matrix and distortion coefficients and save it
 * input: frame -> contain current frame information
 * input: point_list -> list of chessboard corner points in world coordinates
 * input: corner_list -> list of chessboard corner points as found in the image
 * input: image_list -> list of image names used for calibration, used to save the rvec and tvec associated with each images
 * input: saveDir -> saving directory for the camera intrinstic parameter
 */
int calibrateNSave(Mat &frame, vector<vector<cv::Vec3f>> &point_list, vector<vector<cv::Point2f>> &corner_list, vector<string> &image_list, string saveDir)
{
    try
    {
        // Check for validity of input data
        if (point_list.size() < 5)
        {
            printf("Not enough data for calibration\n");
            return (-1);
        }
        if (point_list.size() != corner_list.size())
        {
            printf("Data error, size of point_list != size of corner_list");
            return (-1);
        }

        // Initialize the camera matrix and distortion coefficient
        Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
        cameraMatrix.at<double>(0, 2) = frame.cols / 2;
        cameraMatrix.at<double>(1, 2) = frame.rows / 2;
        Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
        vector<Mat> rvecs;
        vector<Mat> tvecs;

        // Start calibration
        cout << "camera matrix before calibration " << endl
             << cameraMatrix << endl;
        cout << "dist coefficient before calibration " << endl
             << distCoeffs << endl;
        double reprojectionError = cv::calibrateCamera(point_list, corner_list, frame.size(), cameraMatrix, distCoeffs, rvecs, tvecs, cv::CALIB_FIX_ASPECT_RATIO);
        cout << "camera matrix after calibration " << endl
             << cameraMatrix << endl;
        cout << "dist coefficient after calibration " << endl
             << distCoeffs << endl;
        cout << "Re-projection error reported by calibrateCamera: " << reprojectionError << endl;

        // Save the result to file
        if (saveIntrinsic(cameraMatrix, distCoeffs, saveDir) != 0)
        {
            return (-1);
        };
        // Save rvec and tvec for each images
        for (int i = 0; i < rvecs.size(); i++)
        {
            saveImgProp(rvecs[i], tvecs[i], image_list[i]);
        }
    }
    catch (exception)
    {
        cout << "Error in calibration" << endl;
        return (-1);
    }

    return 0;
}

/**
 * save camera intrinsic into the xml file
 * input: Mat of cameraMatrix
 * input: Mat of distortion coefficient
 * input: save directory
 */
int saveIntrinsic(Mat &cameraMatrix, Mat &distCoeffs, string &saveDir)
{
    try
    {
        string filename = saveDir + "cameraIntrinsic.xml";
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        fs << "CameraMatrix" << cameraMatrix;
        fs << "DistortionCoeffs" << distCoeffs;
        fs.release();
        cout << "Write Done." << endl;
    }
    catch (exception)
    {
        cout << "Error in saving" << endl;
        return (-1);
    }

    return 0;
}

/**
 * save image properties into the xml file
 * input: Mat of rvec
 * input: Mat of tvec
 * input: save file name
 */
int saveImgProp(cv::Mat &rvec, cv::Mat &tvec, std::string &fileName)
{
    try
    {
        string filename = fileName + ".xml";
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        fs << "rvec" << rvec;
        fs << "tvec" << tvec;
        fs.release();
    }
    catch (exception)
    {
        cout << "Error in saving" << endl;
        return (-1);
    }

    return 0;
}

/**
 * read camera intrinsic from the xml file
 * Output: Mat of cameraMatrix
 * Output: Mat of distortion coefficient
 * input: save directory
 */
int readIntrinsic(cv::Mat &cameraMatrix, cv::Mat &distCoeffs, std::string &saveDir)
{
    try
    {
        string filename = saveDir + "cameraIntrinsic.xml";
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        fs["CameraMatrix"] >> cameraMatrix;
        fs["DistortionCoeffs"] >> distCoeffs;
        fs.release();
        cout << "Reading Done." << endl;
        cout << "Camera Matrix" << endl
             << cameraMatrix << endl;
        cout << "Distortion Coeffs" << endl
             << distCoeffs << endl;
    }
    catch (exception)
    {
        cout << "Error in saving" << endl;
        return (-1);
    }

    return 0;
}

/**
 * Helper function for getOptionParam, check if the option is the prefix of the target string
 * Input : prefix string (option looking for)
 * Input : target string
 */
bool isPrefix(const std::string &prefix, const std::string &target)
{
    if (prefix.length() > target.length())
    {
        return false;
    }
    return target.compare(0, prefix.length(), prefix) == 0;
}

/**
 * Helper function to retrieve the parameter of associated option
 * Input : argc (number of argument)
 * Input : char array of argv (as passed from the command line)
 * Input : option (string of the target option, example "-row=")
 */
std::string getOptionParam(int argc, char *argv[], const std::string &option)
{
    std::string optionParam;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (isPrefix(option, arg))
        {
            optionParam = arg.substr(option.length());
        }
    }
    return optionParam;
}

/**
 * Overlay the warpImg onto the destination
 * Input : Mat of warpImg
 * Output: Mat of dst (destination frame)
 * Input : Vector of corner points (used to save computation time)
 */
int overlayImg(cv::Mat &warpedImg, cv::Mat &dst, std::vector<cv::Point2f> &cornerInImgs)
{
    try
    {
        // To save computation time, get the border using the corners
        int leftBorder = cornerInImgs[0].x;
        int rightBorder = cornerInImgs[1].x;
        int topBorder = cornerInImgs[0].y;
        int btmBorder = cornerInImgs[2].y;
        for (auto corner : cornerInImgs)
        {
            leftBorder = std::min<int>(leftBorder, (int)corner.x);
            rightBorder = std::max<int>(rightBorder, (int)corner.x);
            topBorder = std::min<int>(topBorder, (int)corner.y);
            btmBorder = std::max<int>(btmBorder, (int)corner.y);
        }
        // Ensure the borders dont go out of bound
        leftBorder = std::max<int>(leftBorder, 0);
        rightBorder = std::min<int>(rightBorder, dst.cols - 1);
        topBorder = std::max<int>(topBorder, 0);
        btmBorder = std::min<int>(btmBorder, dst.rows - 1);
        // Check the pixels in warped image, if it is not black, update it to display frame
        for (int row = topBorder; row <= btmBorder; row++)
        {
            for (int col = leftBorder; col <= rightBorder; col++)
            {
                Vec3b warpedpt = warpedImg.at<Vec3b>(row, col);
                if (warpedpt != Vec3b(0, 0, 0))
                {
                    dst.at<Vec3b>(row, col) = warpedpt;
                }
            }
        }
    }
    catch (exception)
    {
        cout << "Error in overlay" << endl;
        return (-1);
    }

    return 0;
}