/*
  Shiang Jin, Chin

  All calculators to convert given image into a set of vectors for project 2
 */

#ifndef FEATURECALCS_H
#define FEATURECALCS_H

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <ctime>
#include "filters.h"
using namespace std;
using namespace cv;

/**
 * Calculate baseline feature vectors and store in the list
 * by extracting the 9 x 9 square in the middle of the image
 */
int calcBaseline(cv::Mat &src, vector<int> &featureVectors);

/**
 * Calculate the feature vector based on rg chromaticity histogram
 * Output the normalized histogram values into the feature vectors
 * the histSize can be set by the user
 */
int calcRGHist(cv::Mat &src, vector<float> &featureVectors, const int histSize);

/**
 * Calculate the feature vector based on R, G, B histogram of 8 bins each
 * flatten the normalized histogram values into the 1-D feature vectors
 */
int calcRGBHist(cv::Mat &src, vector<float> &featureVectors);

/**
 * Calculate the feature vector based on two histogram (left and right of the image)
 * Output the normalized histogram values into the feature vectors
 * the histSize can be set by the user
 */
int calcMultiHistLR(cv::Mat &src, vector<float> &featureVectors, const int histSize);

/**
 * Calculate the feature vector based on a whole image color histogram
 * and a whole image texture histogram based on gradient magnitude image
 */
int calcRGBNTexture(cv::Mat &src, vector<float> &featureVectors);

/**
 * Calculate the feature vector based on a zoom portion of the image
 * combining color histogram and texture histogram based on gradient magnitude image
 * acceptable zoomFactor is between 0.1 to 1.0 (whole image)
 * the zoomed image will center around the center of the original image
 */
int calcRGBNTexture(cv::Mat &src, vector<float> &featureVectors, const float zoomFactor);

/*
 * Calculate the feature vector based on a whole image color histogram
 * and a whole image texture histogram based on output image after applying a set of gabor filters
 */
int calcRGBNGabor(cv::Mat &src, vector<float> &featureVectors, bool smallImg);

/**
 * Calculate the feature vector based on a zoom portion of the image
 * combining color histogram and texture histogram based on gabor filters
 * acceptable zoomFactor is between 0.1 to 1.0 (whole image)
 * the zoomed image will center around the center of the original image
 */
int calcRGBNGabor(cv::Mat &src, vector<float> &featureVectors, const float zoomFactor);
#endif
