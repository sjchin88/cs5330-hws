/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/06/2023
  Description   : All required method to search the database
*/

#ifndef SEARCHDB_H
#define SEARCHDB_H
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include "featureCalcs.h"
#include "csv_util.h"
#include "distMetrics.h"

/**
 * Struct to store the result
 */
struct ResultStruct
{
  char *imgName;
  float distance;
  ResultStruct(char *name, float dist) : imgName(name), distance(dist) {}
  bool operator<(const ResultStruct &result) const
  {
    return (distance < result.distance);
  }
};

/**
 * compute the distance in baseline database located in csvFilePath
 * compare to the targetImg
 * store all the result in resultList
 * For baseline, default is using sum of square distance method
 */
int searchBaseline(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList);

/**
 * compute the distance in rg histogram database located in csvFilePath
 * compare to the targetImg
 * store all the result in resultList
 * histSize determine the size of n-bins used
 * distIdx determine which dist metrics to use
 * 1 - for sum of square, 2 - for histogram intersection
 */
int searchRGHist(cv::Mat &targetImg, string &csvFilePath, const int histSize, vector<ResultStruct> &resultList, const int distIdx);

/**
 * compute the distance in multiHistogram (with Left & Right part) database located in csvFilePath
 * compare to the targetImg
 * store all the result in resultList
 * histSize determine the size of n-bins used
 * distIdx determine which dist metrics to use
 * 1 - for sum of square, 2 - for histogram intersection
 */
int searchMultiHistLR(cv::Mat &targetImg, string &csvFilePath, const int histSize, vector<ResultStruct> &resultList, const int distIdx);

/**
 * compute the distance of images in the RGB and (gradient magnitude) Texture combined histogram database located in csvFilePath
 * compare to the targetImg
 * store all the result in resultList
 * distIdx determine which dist metrics to use
 * 1 - for sum of square, 2 - for histogram intersection
 * zoom factor determine portion of image to focus around center
 * default value is 1.0 (whole image)
 * note the zoom factor need to be the same as value used in building the image database
 */
int searchRGBNTexture(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList, const int distIdx = 2, float zoomFactor = 1.0F);

/**
 * compute the distance of images in the RGB and Gabor Texture combined histogram database located in csvFilePath
 * compare to the targetImg
 * store all the result in resultList
 * distIdx determine which dist metrics to use
 * 1 - for sum of square, 2 - for histogram intersection
 * zoom factor determine portion of image to focus around center
 * default value is 1.0 (whole image)
 * note the zoom factor need to be the same as value used in building the image database
 */
int searchRGBNGabor(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList, const int distIdx = 2, float zoomFactor = 1.0F);

#endif