/*
  Shiang Jin, Chin

  All distMetrics to compute the distance between two vectors
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
 */
int histogram_intersect(vector<float> &targetFeature, vector<float> &srcFeature, float &result);

#endif