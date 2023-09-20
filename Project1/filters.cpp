#include "filters.h";

void printTest(int input)
{
    cout << "print test";
}

int greyscale(cv::Mat &src, cv::Mat &dst)
{
    const int rows = src.rows;
    const int cols = src.cols;
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            Vec3b bgrPixel = src.at<cv::Vec3b>(y, x); // gives you the BGR vector of type cv::Vec3band will be in row, column order
            // bgrPixel[0]= Blue//
            // bgrPixel[1]= Green//
            // bgrPixel[2]= Red//
            int grey_intensity = 0.1 * bgrPixel[0] + 0.8 * bgrPixel[1] + 0.1 * bgrPixel[2];
            dst.at<uchar>(y, x) = grey_intensity;
        }
    }
    return 0;
}