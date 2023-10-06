#ifndef FILTER_H
#define FILTER_H
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <ctime>
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

/**
 * Create a gabor filtered images based on gabor filters parameters of
 * ksize and
 * sigmas, thetas, lambdas and gammas,
 * note these four parameters are loops thus if you have
 * 2 of each, total of 16 gabor filter will be created and used to filter the image
 */
int gaborFiltering(cv::Mat &src, cv::Mat &dst, cv::Size &ksize, vector<float> &sigmas, vector<float> &thetas, vector<float> &lambdas, vector<float> &gammas);
#endif