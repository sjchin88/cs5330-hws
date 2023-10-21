/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/06/2023
  Description   : All calculators to convert given image into a set of feature vectors for project 2
*/
#ifndef FEATURECALCS_H
#define FEATURECALCS_H

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "filters.h"
using namespace std;
using namespace cv;

/**
 * Calculate baseline feature vectors from the src and append them into the featureVectors
 * by extracting the 9 x 9 square in the middle of the image
 */
int calcBaseline(cv::Mat &src, vector<int> &featureVectors);

/**
 * Calculate the feature vectors from the src and append them into the featureVectors
 * based on rg chromaticity histogram
 * the histSize can be set by the user
 */
int calcRGHist(cv::Mat &src, vector<float> &featureVectors, const int histSize);

/**
 * Calculate the feature vectors based on R, G, B histogram of 8 bins each frpm the src
 * flatten the normalized histogram values into the 1-D feature vectors
 * and append them into the featureVectors
 */
int calcRGBHist(cv::Mat &src, vector<float> &featureVectors);

/**
 * Calculate the feature vectors based on two rg chromaticity histogram
 * for the left and right part of the src image
 * Output the normalized histogram values into the feature vectors
 * the histSize can be set by the user
 */
int calcMultiHistLR(cv::Mat &src, vector<float> &featureVectors, const int histSize);

/**
 * Calculate the feature vectors based on a zoom portion of the src image
 * combining color histogram and texture histogram based on gradient magnitude image
 * acceptable zoomFactor is between 0.1 to 1.0 (1.0 is default option and cover whole image)
 * the zoomed image will center around the center of the original image
 * append them into the featureVectors
 */
int calcRGBNTexture(cv::Mat &src, vector<float> &featureVectors, float zoomFactor = 1.0F);

/**
 * Calculate the feature vector based on a zoom portion of the src image
 * combining color histogram and texture histogram based on gabor filters
 * acceptable zoomFactor is between 0.1 to 1.0 (1.0 is default option and cover whole image)
 * the zoomed image will center around the center of the original image
 * append them into the featureVectors
 */
int calcRGBNGabor(cv::Mat &src, vector<float> &featureVectors, float zoomFactor = 1.0F);
#endif
