/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/06/2023
  Description   : Filters from project 1&2
                * Filters used in project 3:

*/
#ifndef FILTER_H
#define FILTER_H
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <ctime>
#include "utils.h"
using namespace std;
using namespace cv;

/**
 * Alternative greyscale based on custom constants vector
 * Input : src image of BGR color in 3 channels
 * Input : constants factor for BGR for grayscale conversion
 * Output: dst image of grayscale in 1 channels
 */
int greyscale(cv::Mat &src, vector<float> constants, cv::Mat &dst);

/**
 * 5x5 Gaussian filter
 */
int blur5x5(cv::Mat &src, cv::Mat &dst);

/**
 * Sobel X , positive right
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

/**
 * Sobel Y , positive top
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

/**
 * sobel 3x3 helper function
 */
int sobel3x3(cv::Mat &src, cv::Mat &dst, int vectorX[3], int vectorY[3]);

/**
 * magnitude filter based on SobelX and SobelY
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

/**
 * blurs a color image using the 5x5 Gaussian filter
 * then quantizes it
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

/**
 * cartonize the imgage by
 * using the gradient magnitude value
 * to determine whether to keep the blur&Quantize value
 */
int cartoon(cv::Mat &src, cv::Mat &dst, int levels, int magThreshold);

/**
 * turn the image into negative of itself
 */
int negative(cv::Mat &src, cv::Mat &dst);

/**
 * Create a gabor filtered images based on gabor filters parameters of
 * ksize and
 * sigmas, thetas, lambdas and gammas,
 * note these four parameters are loops thus if you have
 * 2 of each, total of 2 x 2 x 2 x 2 = 16 gabor filter will be created and used to filter the image
 */
int gaborFiltering(cv::Mat &src, cv::Mat &dst, cv::Size &ksize, vector<float> &sigmas, vector<float> &thetas, vector<float> &lambdas, vector<float> &gammas);

/**
 * Apply threshold to separates an object from the background
 */
int thresholdFilter(cv::Mat &src, cv::Mat &dst);

/**
 * Apply the morphologyFilter with preset setting
 */
int morphologyFilter(cv::Mat &src, cv::Mat &dst);

#endif