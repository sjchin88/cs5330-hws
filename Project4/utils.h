/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   :
*/

#ifndef UTIL_H
#define UTIL_H
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <string>

int calibrateNSave(cv::Mat &frame, std::vector<std::vector<cv::Vec3f>> &point_list, std::vector<std::vector<cv::Point2f>> &corner_list, std::string saveDir);
int saveIntrinsic(cv::Mat &cameraMatrix, cv::Mat distCoeffs, std::string saveDir);
int readIntrinsic(cv::Mat &cameraMatrix, cv::Mat distCoeffs, std::string saveDir);

#endif