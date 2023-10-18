/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   : Contain all required methods other than filters for Project 3
*/
#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
using namespace cv;
using namespace std;
/**
 * Return a vector of threshold values use to distinguish
 * foreground and background
 * Input: src image of grayscale in one channel
 * Output: single threshold value
 */
int getThreshold(cv::Mat &src, vector<int> &threshold);

/**
 * Retrieve the centers calculated using kmeans algorithm
 * of two clusters
 */
int getKMeansCenters(cv::Mat &src, cv::Mat &centers);

/**
 * Get the connected components using standard openCV connectedComponentsWithStats function
 * Input: Mat of src image
 * Output: List of connected components' region map
 * flag: showCC default is false, if set to true, will Labeled the connected region with same color
 * and show it in a window
 */
int getConnectedComponentRegions(cv::Mat &src, vector<cv::Mat> &regionList, bool showCC = false);

/**
 * Using the rotated bounding rectangle with minimum area obtained from region of interest
 * calculate the scale, translational, and rotational invariant for
 * percentage filled, and bounding rectangle width/height
 * Input: Mat of src image
 * Output: the two features computed appended to the features list
 * flag: showAxis default is false, if set to true, will draw the axis for least 2nd moment for the object
 */
int getOrientedBoundingBoxStat(cv::Mat &src, vector<float> &features, bool showAxis = false);
#endif