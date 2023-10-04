#ifndef FILTER_H
#define FILTER_H
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int greyscale(cv::Mat &src, cv::Mat &dst);
int blur5x5(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int sobel3x3(cv::Mat &src, cv::Mat &dst, int vectorX[3], int vectorY[3]);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);
int cartoon(cv::Mat &src, cv::Mat &dst, int levels, int magThreshold);
int negative(cv::Mat &src, cv::Mat &dst);
#endif