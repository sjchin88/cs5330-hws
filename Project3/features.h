/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   : Contain all required methods to compute the features for Project 3
*/
#ifndef FEATURES_H
#define FEATURES_H
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include <math.h>
#include "filters.h"
#include "file_util.h"

#define PI 3.14159265
using namespace cv;
using namespace std;

// enum for common feature type
enum featuresTypes
{
  F_PERCENT_FILLED = 0,
  F_WIDTH_HEIGHT_Ratio = 1,
  F_HU_1ST_MOMENT = 2,
  F_HU_2ND_MOMENT = 3,
  F_HU_3RD_MOMENT = 4,
  F_HU_4TH_MOMENT = 5,
  F_HU_5TH_MOMENT = 6,
  F_HU_6TH_MOMENT = 7,
  F_HU_7TH_MOMENT = 8,
  F_CENTROID_X = 9,
  F_CENTROID_Y = 10,
  F_FEATURES_SIZE = 11
};

/**
 * Struct to store the region stats with label from connectedComponentWithStat
 * Used to sort the region and color only the topN largest region
 */
struct RegionStruct
{
  int regionId;
  cv::Mat regionStat;
  RegionStruct(int label, cv::Mat &stat) : regionId(label), regionStat(stat.clone()) {}
  bool operator<(const RegionStruct &region) const
  {
    return (regionStat.at<int>(CC_STAT_AREA) > region.regionStat.at<int>(CC_STAT_AREA));
  }
};

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
 * input : topN, return only the top N largest region found, default is 1
 */
int getConnectedComponentRegions(cv::Mat &src, vector<cv::Mat> &regionList, bool showCC = false, int topN = 1);

/**
 * Compute the scale, translational, and rotational invariant features,
 * first use the rotated bounding rectangle with minimum area obtained from region of interest to calculate for
 * percentage filled, and bounding rectangle width/height
 * Next use the cv::moments() and cv::HuMoments to get the 7 variants of Hu moments
 * Input: Mat of src image , after thresholding and clean up
 * Input: regionStatList for major region identified
 * Output: the nine features computed for each region appended to the featuresList
 * flag: showAR, default is false, if set to true, will show the rotated bounding rectangle and
 * axis of least moment
 */
int computeFeatures(cv::Mat &src, vector<cv::Mat> regionStatList, vector<vector<double>> &features, bool showAR = false);

/**
 * Main function to process the image to detect the object
 * and compute the feature vectors for the object
 * input : Mat of src image
 * Output: Vector of features for each component
 * input : topN, return only the top N largest region found, default is 1
 * flag  : showInterim, default=false, if it is true, will show images after each preprocessing
 * flag  : showThreshold, default=false, if it is true, will show images after thresholding
 * flag  : showMorpho, default=false, if it is true, will show images after morphological operation
 * flag  : showCC, default=false, if it is true, will show the colored connected components
 * flag  : showAR, default=false, if it is true, will show the axis for least 2nd moment and rotated bounding rectangle of minimum area
 */
int processNCompute(cv::Mat &src, vector<vector<double>> &features, int topN = 1, bool showInterim = false, bool showThreshold = false, bool showMorpho = false, bool showCC = false, bool showAR = false);

/**
 * Helper method to extra the feature texts
 * input : vector of computed features
 * output: string text containing all features
 */
int getFeatureText(vector<double> &features, vector<string> &texts);
#endif