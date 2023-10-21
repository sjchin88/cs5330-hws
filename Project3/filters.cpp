/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   : Filters required for project 3
*/
#include "filters.h";

using namespace cv;
using namespace std;

/**
 * Alternative greyscale based on custom constants vector
 * Input : src image of BGR color in 3 channels
 * Input : constants factor for BGR for grayscale conversion
 * Output: dst image of grayscale in 1 channels
 */
int greyscale(cv::Mat &src, vector<float> constants, cv::Mat &dst)
{
    try
    {
        // Get the number of rows and cols from the source
        const int rows = src.rows;
        const int cols = src.cols;
        float bConst = constants[0];
        float gConst = constants[1];
        float rConst = constants[2];
        Mat tempFrame(src.size(), CV_8UC1, Scalar(0));
        // Looping through each pixels for the rows and cols
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                // Get the BGR vector of individual pixel of type cv::Vec3b
                // bgrPixel[0]= Blue//
                // bgrPixel[1]= Green//
                // bgrPixel[2]= Red//
                Vec3b bgrPixel = src.at<cv::Vec3b>(y, x);

                // Custom processing function for the individual pixel
                int grey_intensity = bConst * bgrPixel[0] + gConst * bgrPixel[1] + rConst * bgrPixel[2];

                tempFrame.at<uint8_t>(y, x) = grey_intensity;
            }
        }
        dst = tempFrame;
        return 0;
    }
    catch (std::exception)
    {
        return 1;
    }
}

// Task 5: 5x5 Gaussian filter
int blur5x5(cv::Mat &src, cv::Mat &dst)
{
    try
    {
        // Get the number of rows and cols from the source
        const int rows = src.rows;
        const int cols = src.cols;
        int factors[5] = {1, 2, 4, 2, 1};

        // First transform, horizontally
        Mat tempFrame1(src.size(), CV_8UC3, Scalar(0));
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                int blue = 0;
                int green = 0;
                int red = 0;
                int totalFactor = 0;
                // use leftLimit & rightLimit to handle edges
                int leftLimit = max(0, x - 2);
                int rightLimit = min(cols - 1, x + 2);
                for (int col = leftLimit; col <= rightLimit; col++)
                {
                    // if col = 0 and x = 0, the factor will be factors[2]
                    // which is the center location of x itself
                    int factor = factors[col - x + 2];
                    Vec3b temp = src.at<cv::Vec3b>(y, col);
                    blue += temp[0] * factor;
                    green += temp[1] * factor;
                    red += temp[2] * factor;
                    totalFactor += factor;
                }
                tempFrame1.at<cv::Vec3b>(y, x)[0] = int(blue / totalFactor);
                tempFrame1.at<cv::Vec3b>(y, x)[1] = int(green / totalFactor);
                tempFrame1.at<cv::Vec3b>(y, x)[2] = int(red / totalFactor);
            }
        }

        // Second transform, vertically
        Mat tempFrame2(src.size(), CV_8UC3, Scalar(0));
        for (int x = 0; x < cols; x++)
        {
            for (int y = 0; y < rows; y++)
            {
                int blue = 0;
                int green = 0;
                int red = 0;
                int totalFactor = 0;
                // use topLimit & btmLimit to handle edges
                int topLimit = max(0, y - 2);
                int btmLimit = min(rows - 1, y + 2);
                for (int row = topLimit; row <= btmLimit; row++)
                {
                    // if row = 0 and y = 0, the factor will be factors[2]
                    // which is the center location of y itself
                    int factor = factors[row - y + 2];
                    Vec3b temp = tempFrame1.at<cv::Vec3b>(row, x);
                    blue += temp[0] * factor;
                    green += temp[1] * factor;
                    red += temp[2] * factor;
                    totalFactor += factor;
                }
                tempFrame2.at<cv::Vec3b>(y, x)[0] = int(blue / totalFactor);
                tempFrame2.at<cv::Vec3b>(y, x)[1] = int(green / totalFactor);
                tempFrame2.at<cv::Vec3b>(y, x)[2] = int(red / totalFactor);
            }
        }

        dst = tempFrame2;

        return (0);
    }
    catch (std::exception)
    {
        return (-1);
    }
}

/**
 * Helper function to decrease the lightness based on saturation level
 */
int decreaseLightness(cv::Mat &src, cv::Mat &dst)
{
    try
    {
        // Step 1, convert to HSL, store in temp
        Mat temp;
        cv::cvtColor(src, temp, cv::COLOR_BGR2HLS);
        // Loop through the pixels and set the lightness to be L/(1 + S/255)
        for (int y = 0; y < temp.rows; y++)
        {
            for (int x = 0; x < temp.cols; x++)
            {
                Vec3b hlsPixel = temp.at<cv::Vec3b>(y, x);
                int lightness = hlsPixel[1];
                int saturation = hlsPixel[2];
                temp.at<cv::Vec3b>(y, x)[1] = (int)(lightness / (1 + saturation * 1.0 / 255));
            }
        }

        // convert back to BGR
        cv::cvtColor(temp, dst, cv::COLOR_HLS2BGR);
        // imshow("decrease lightness", dst);
    }
    catch (exception)
    {
        return (-1);
    }
    return 0;
}

/**
 * Apply threshold to separates an object from the background
 * input : Mat of src frame
 * Output: Mat of dst frame
 * flag  : showInterim default is false, if set to true will show all frame after each preprocessing
 */
int thresholdFilter(cv::Mat &src, cv::Mat &dst, bool showInterim)
{
    try
    {
        // Preprocessing 1 Gaussian Blur
        Mat temp1;
        if (blur5x5(src, temp1) != 0)
        {
            return (-1);
        }
        if (showInterim)
        {
            imshow("After Preprocessing: Blur", temp1);
        }

        // Preprocessing 2 Decrease lightness based on saturation level
        Mat temp2;
        if (decreaseLightness(temp1, temp2) != 0)
        {
            return (-1);
        }
        if (showInterim)
        {
            imshow("After Preprocessing: Decrease lightness", temp2);
        }

        // Convert the RGB image to grayscale based on custom formula
        vector<float> constants = {0.33, 0.33, 0.34};
        Mat tempGray;
        if (greyscale(temp2, constants, tempGray) != 0)
        {
            return (-1);
        }
        if (showInterim)
        {
            imshow("Converted to grayscale", tempGray);
        }
        // Get the thresholds
        vector<int> thresholds;
        if (getThreshold(tempGray, thresholds) != 0)
        {
            return (-1);
        };
        // cout << "threshold size and value" << thresholds.size() << ":" << thresholds[0] << endl;
        // Loop through the tempGray pixels and set the dst values according to threshold
        dst = Mat::zeros(tempGray.size(), CV_8UC1);
        for (int y = 0; y < tempGray.rows; y++)
        {
            for (int x = 0; x < tempGray.cols; x++)
            {
                dst.at<uint8_t>(y, x) = tempGray.at<uint8_t>(y, x) < thresholds[0] ? 255 : 0;
            }
        }
    }
    catch (exception)
    {
        return (-1);
    }
    return (0);
}

/**
 * Apply the morphologyFilter with preset setting
 */
int morphologyFilter(cv::Mat &src, cv::Mat &dst)
{
    try
    {
        Mat kernel = cv::Mat::ones(cv::Size(5, 5), CV_8UC1);
        cv::morphologyEx(src, dst, MORPH_OPEN, kernel);
    }
    catch (exception)
    {
        return (-1);
    }
    return 0;
}