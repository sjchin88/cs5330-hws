/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   : Filters required for project 3

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
#include "file_util.h"
#include "features.h"
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
 * Apply threshold to separates an object from the background
 * input : Mat of src frame
 * Output: Mat of dst frame
 * flag  : showInterim default is false, if set to true will show all frame after each preprocessing
 */
int thresholdFilter(cv::Mat &src, cv::Mat &dst, bool showInterim = false);

/**
 * Apply the morphologyFilter with preset setting
 */
int morphologyFilter(cv::Mat &src, cv::Mat &dst);

#endif