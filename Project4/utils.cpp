#include "utils.h"

using namespace std;
using namespace cv;

int calibrateNSave(Mat &frame, vector<vector<cv::Vec3f>> &point_list, vector<vector<cv::Point2f>> &corner_list, string saveDir)
{
    try
    {
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
        Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
        cameraMatrix.at<double>(0, 2) = frame.cols / 2;
        cameraMatrix.at<double>(1, 2) = frame.rows / 2;
        Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
        vector<Mat> rvecs;
        vector<Mat> tvecs;
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
        if (saveIntrinsic(cameraMatrix, distCoeffs, saveDir) != 0)
        {
            return (-1);
        };
    }
    catch (exception)
    {
        cout << "Error in calibration" << endl;
        return (-1);
    }

    return 0;
}

int saveIntrinsic(Mat &cameraMatrix, Mat distCoeffs, string saveDir)
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

int readIntrinsic(Mat &cameraMatrix, Mat distCoeffs, string saveDir)
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