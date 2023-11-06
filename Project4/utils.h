/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 11/3/2023
  Description   : Utility functions required for Project 4
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

/**
 * Function to calibrate the camera intrinsic parameters, get the camera matrix and distortion coefficients and save it
 * input: frame -> contain current frame information
 * input: point_list -> list of chessboard corner points in world coordinates
 * input: corner_list -> list of chessboard corner points as found in the image
 * input: image_list -> list of image names used for calibration, used to save the rvec and tvec associated with each images
 * input: saveDir -> saving directory for the camera intrinstic parameter
 */
int calibrateNSave(cv::Mat &frame, std::vector<std::vector<cv::Vec3f>> &point_list, std::vector<std::vector<cv::Point2f>> &corner_list, std::vector<std::string> &image_list, std::string saveDir);

/**
 * save camera intrinsic into the xml file
 * input: Mat of cameraMatrix
 * input: Mat of distortion coefficient
 * input: save directory
 */
int saveIntrinsic(cv::Mat &cameraMatrix, cv::Mat &distCoeffs, std::string &saveDir);

/**
 * save camera intrinsic into the xml file
 * input: Mat of rvec
 * input: Mat of tvec
 * input: save file name
 */
int saveImgProp(cv::Mat &rvec, cv::Mat &tvec, std::string &fileName);

/**
 * read camera intrinsic from the xml file
 * Output: Mat of cameraMatrix
 * Output: Mat of distortion coefficient
 * input: save directory
 */
int readIntrinsic(cv::Mat &cameraMatrix, cv::Mat &distCoeffs, std::string &saveDir);

/**
 * Helper function to retrieve the parameter of associated option
 * Input : argc (number of argument)
 * Input : char array of argv (as passed from the command line)
 * Input : option (string of the target option, example "-row=")
 */
std::string getOptionParam(int argc, char *argv[], const std::string &option);

/**
 * Overlay the warpImg onto the destination
 * Input : Mat of warpImg
 * Output: Mat of dst (destination frame)
 * Input : Vector of corner points (used to save computation time)
 */
int overlayImg(cv::Mat &warpedImg, cv::Mat &dst, std::vector<cv::Point2f> &cornerInImgs);
#endif