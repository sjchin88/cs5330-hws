/*
  Shiang Jin, Chin

  All required method to search the database
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
 */
int searchBaseline(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList);

/**
 * compute the distance in rg histogram database located in csvFilePath
 * compare to the targetImg
 * store all the result in resultList
 * histSize determine the size of n-bins used
 */
int searchRGHist(cv::Mat &targetImg, string &csvFilePath, const int histSize, vector<ResultStruct> &resultList);

/**
 * compute the distance in multiHistogram (with Left & Right part) database located in csvFilePath
 * compare to the targetImg
 * store all the result in resultList
 * histSize determine the size of n-bins used
 */
int searchMultiHistLR(cv::Mat &targetImg, string &csvFilePath, const int histSize, vector<ResultStruct> &resultList);

/**
 * compute the distance of images in the RGB and (gradient magnitude) Texture combined histogram database located in csvFilePath
 * compare to the targetImg
 * store all the result in resultList
 */
int searchRGBNTexture(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList);

/**
 * compute the distance of images in the RGB and (gradient magnitude) Texture combined histogram database located in csvFilePath
 * compare to the targetImg with specified zoomFactor
 * store all the result in resultList
 */
int searchRGBNTexture(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList, float zoomFactor);

/**
 * compute the distance of images in the RGB and Gabor Texture combined histogram database located in csvFilePath
 * compare to the targetImg
 * store all the result in resultList
 */
int searchRGBNGabor(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList);

/**
 * compute the distance of images in the RGB and Gabor Texture combined histogram database located in csvFilePath
 * compare to the targetImg with specified zoomFactor
 * store all the result in resultList
 */
int searchRGBNGabor(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList, float zoomFactor);
#endif