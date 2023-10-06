#include "filters.h";

using namespace cv;
using namespace std;

// Task 4 : Alternative greyscale version
int greyscale(cv::Mat &src, cv::Mat &dst)
{
    try
    {
        // Get the number of rows and cols from the source
        const int rows = src.rows;
        const int cols = src.cols;
        Mat tempFrame(src.size(), CV_8UC3, Scalar(0));
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
                int grey_intensity = 0.1 * bgrPixel[0] + 0.1 * bgrPixel[1] + 0.8 * bgrPixel[2];

                tempFrame.at<cv::Vec3b>(y, x)[0] = grey_intensity;
                tempFrame.at<cv::Vec3b>(y, x)[1] = grey_intensity;
                tempFrame.at<cv::Vec3b>(y, x)[2] = grey_intensity;
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

        return 0;
    }
    catch (std::exception)
    {
        return 1;
    }
}

// Task 6: Sobel X , positive right
int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    int vectorX[3] = {-1, 0, 1};
    int vectorY[3] = {1, 2, 1};
    return sobel3x3(src, dst, vectorX, vectorY);
}

// Task 6: Sobel Y , positive top
int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    int vectorX[3] = {1, 2, 1};
    int vectorY[3] = {1, 0, -1};
    return sobel3x3(src, dst, vectorX, vectorY);
}

// Helper function for Task 6
int sobel3x3(cv::Mat &src, cv::Mat &dst, int vectorX[3], int vectorY[3])
{
    try
    {
        // Get the number of rows and cols from the source
        const int rows = src.rows;
        const int cols = src.cols;

        // First transform, horizontally
        Mat tempFrame1(src.size(), CV_16SC3, Scalar(0));
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                int blue = 0;
                int green = 0;
                int red = 0;
                // use leftLimit & rightLimit to handle edges
                int leftLimit = max(0, x - 1);
                int rightLimit = min(cols - 1, x + 1);
                for (int col = leftLimit; col <= rightLimit; col++)
                {
                    // if col = 0 and x = 0, the factor will be factors[1]
                    // which is the center location of x itself
                    int factor = vectorX[col - x + 1];
                    Vec3b temp = src.at<cv::Vec3b>(y, col);
                    blue += temp[0] * factor;
                    green += temp[1] * factor;
                    red += temp[2] * factor;
                    // totalFactor += factor;
                }
                tempFrame1.at<cv::Vec3s>(y, x)[0] = blue;
                tempFrame1.at<cv::Vec3s>(y, x)[1] = green;
                tempFrame1.at<cv::Vec3s>(y, x)[2] = red;
            }
        }

        // Second transform, vertically
        Mat tempFrame2(src.size(), CV_16SC3, Scalar(0));
        for (int x = 0; x < cols; x++)
        {
            for (int y = 0; y < rows; y++)
            {
                int blue = 0;
                int green = 0;
                int red = 0;
                // use topLimit & btmLimit to handle edges
                int leftLimit = max(0, y - 1);
                int rightLimit = min(rows - 1, y + 1);
                for (int row = leftLimit; row <= rightLimit; row++)
                {
                    // if row = 0 and y = 0, the factor will be factors[1]
                    // which is the center location of y itself
                    int factor = vectorY[row - y + 1];
                    Vec3s temp = tempFrame1.at<cv::Vec3s>(row, x);
                    blue += temp[0] * factor;
                    green += temp[1] * factor;
                    red += temp[2] * factor;
                }
                tempFrame2.at<cv::Vec3s>(y, x)[0] = blue;
                tempFrame2.at<cv::Vec3s>(y, x)[1] = green;
                tempFrame2.at<cv::Vec3s>(y, x)[2] = red;
            }
        }

        dst = tempFrame2;
        return 0;
    }
    catch (std::exception)
    {
        return 1;
    }
}

// Task 7, magnitude filter
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst)
{
    try
    {
        // Get the number of rows and cols from the source
        const int rows = sx.rows;
        const int cols = sx.cols;

        // Initialize output array
        Mat tempFrame(sx.size(), CV_16SC3, Scalar(0));

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                cv::Vec3s temp_sx = sx.at<cv::Vec3s>(y, x);
                cv::Vec3s temp_sy = sy.at<cv::Vec3s>(y, x);
                tempFrame.at<cv::Vec3s>(y, x)[0] = int(sqrt(temp_sx[0] * temp_sx[0] + temp_sy[0] * temp_sy[0]));
                tempFrame.at<cv::Vec3s>(y, x)[1] = int(sqrt(temp_sx[1] * temp_sx[1] + temp_sy[1] * temp_sy[1]));
                tempFrame.at<cv::Vec3s>(y, x)[2] = int(sqrt(temp_sx[2] * temp_sx[2] + temp_sy[2] * temp_sy[2]));
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

// Task 8 blurs and quantizes a color image
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels)
{
    try
    {
        // Get the number of rows and cols from the source
        const int rows = src.rows;
        const int cols = src.cols;
        Mat tempFrame;

        // try applied Gaussian 5x5 filter first
        if (blur5x5(src, tempFrame) != 0)
        {
            return 1;
        }

        // Continue with quantize
        int b = 255 / levels;

        // Looping through each pixels for the rows and cols
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                // Get the BGR vector of individual pixel of type cv::Vec3b
                // bgrPixel[0]= Blue//
                // bgrPixel[1]= Green//
                // bgrPixel[2]= Red//
                Vec3b bgrPixel = tempFrame.at<cv::Vec3b>(y, x);

                // Custom processing function for the individual pixel
                tempFrame.at<cv::Vec3b>(y, x)[0] = (bgrPixel[0] / b) * b;
                tempFrame.at<cv::Vec3b>(y, x)[1] = (bgrPixel[1] / b) * b;
                tempFrame.at<cv::Vec3b>(y, x)[2] = (bgrPixel[2] / b) * b;
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

// Task 9 cartonize the img
int cartoon(cv::Mat &src, cv::Mat &dst, int levels, int magThreshold)
{
    try
    {
        Mat tempFrameX;
        Mat tempFrameY;
        Mat tempFrameM;
        Mat tempFrameBQ;

        // First call the magnitude filter to get the gradientmagnitude, store in tempFrameM
        if (!(sobelX3x3(src, tempFrameX) == 0 && sobelY3x3(src, tempFrameY) == 0 && magnitude(tempFrameX, tempFrameY, tempFrameM) == 0))
        {
            return 1;
        }

        // Next apply the blur and quantize filter to source, store in tempFrameBQ
        if (blurQuantize(src, tempFrameBQ, levels) != 0)
        {
            return 1;
        }

        // Final transformation, if tempFrameM value > magThreshold, set the tempFrameBQ value to 0 (dark)
        // Get the number of rows and cols from the source
        const int rows = src.rows;
        const int cols = src.cols;
        // Looping through each pixels for the rows and cols
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                // Get the BGR vector of individual pixel of type cv::Vec3b
                // bgrPixel[0]= Blue//
                // bgrPixel[1]= Green//
                // bgrPixel[2]= Red//
                Vec3s magPixel = tempFrameM.at<cv::Vec3s>(y, x);

                // Custom processing function for the individual pixel
                if (magPixel[0] > magThreshold)
                {
                    tempFrameBQ.at<cv::Vec3b>(y, x)[0] = 0;
                }

                if (magPixel[1] > magThreshold)
                {
                    tempFrameBQ.at<cv::Vec3b>(y, x)[1] = 0;
                }
                if (magPixel[2] > magThreshold)
                {
                    tempFrameBQ.at<cv::Vec3b>(y, x)[2] = 0;
                }
            }
        }
        dst = tempFrameBQ;
        return 0;
    }
    catch (std::exception)
    {
        return 1;
    }
};

// Task 10 : turn the image into negative of itself
int negative(cv::Mat &src, cv::Mat &dst)
{
    try
    {
        // Get the number of rows and cols from the source
        const int rows = src.rows;
        const int cols = src.cols;
        Mat tempFrame(src.size(), CV_8UC3, Scalar(0));
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
                tempFrame.at<cv::Vec3b>(y, x)[0] = 255 - bgrPixel[0];
                tempFrame.at<cv::Vec3b>(y, x)[1] = 255 - bgrPixel[1];
                tempFrame.at<cv::Vec3b>(y, x)[2] = 255 - bgrPixel[2];
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

/**
 * Helper function to get all combinations of the gabor filters
 */
int buildFilter(cv::Size &ksize, vector<float> &sigmas, vector<float> &thetas, vector<float> &lambdas, vector<float> &gammas, vector<cv::Mat> &gaborKernels)
{
    try
    {
        for (float sigma : sigmas)
        {
            for (float theta : thetas)
            {
                for (float lambda : lambdas)
                {
                    for (float gamma : gammas)
                    {
                        cv::Mat gaborKernel = cv::getGaborKernel(ksize, sigma, theta, lambda, gamma, 0);
                        gaborKernel = gaborKernel / (1.5 * cv::sum(gaborKernel)); // Normalize the gaborKernel
                        gaborKernels.push_back(gaborKernel);
                    }
                }
            }
        }
    }
    catch (exception)
    {
        return (-1);
    }
    return 0;
}
/**
 * Create a gabor filtered images based on gabor filters parameters of
 * ksize and
 * sigmas, thetas, lambdas and gammas,
 * note these four parameters are loops thus if you have
 * 2 of each, total of 16 gabor filter will be created and used to filter the image
 */
int gaborFiltering(cv::Mat &src, cv::Mat &dst, cv::Size &ksize, vector<float> &sigmas, vector<float> &thetas, vector<float> &lambdas, vector<float> &gammas)
{
    try
    {
        vector<cv::Mat> gaborKernels;

        if (buildFilter(ksize, sigmas, thetas, lambdas, gammas, gaborKernels) != 0)
        {
            return (-1);
        }

        // Now process the image
        dst = Mat::zeros(src.size(), CV_8UC3);
        // Loop through each kernel and update the max output
        // from the filter
        cv::Mat tempMat;
        for (Mat kernel : gaborKernels)
        {
            cv::filter2D(src, tempMat, src.depth(), kernel);
            cv::max(dst, tempMat, dst);
        }
    }
    catch (exception)
    {
        return (-1);
    }
    return 0;
}
