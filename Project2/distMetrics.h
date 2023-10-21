/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/06/2023
  Description   : All distMetrics to compute the distance between two vectors
*/
#ifndef DISTMETRICS_H
#define DISTMETRICS_H

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
using namespace std;
using namespace cv;

/**
 * Calculate the sum of square difference between two feature vectors
 */
int sum_of_squared_difference(vector<int> &targetFeature, vector<int> &srcFeature, float &result);

/**
 * Calculate the histogram intersection between two histogram, return as the difference (1 - intersect Total)
 * and stored in result
 */
int histogram_intersect(vector<float> &targetFeature, vector<float> &srcFeature, float &result);

/**
 * Calculate the distance between two vectors
 * Using distOption chosen - 1 for sum_of_squared, 2 - for histogram intersection
 * store the difference in result
 */
int getDistance(vector<float> &targetFeature, vector<float> &srcFeature, float &result, const int distOption);
#endif