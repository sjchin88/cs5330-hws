/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   : Contain all required methods for classification for project 3

*/
#ifndef CLASSIFY_H
#define CLASSIFY_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include <math.h>
#include "filters.h"
#include "file_util.h"
using namespace cv;
using namespace std;

/**
 * Struct to store the object
 */
struct ObjectStruct
{
    char *objName;
    vector<float> features;
    ObjectStruct(char *name, vector<float> &dist) : objName(name), features(dist) {}
};

/**
 * Struct to store the std deviations and avg distance
 */
struct StatStruct
{
    vector<double> stdSquares;
    double avgDist;
    StatStruct(vector<double> &stds, double avg) : stdSquares(stds), avgDist(avg) {}
};

/**
 * Struct to store the result
 */
struct ResultStruct
{
    char *objName;
    double distance;
    ResultStruct(char *name, double dist) : objName(name), distance(dist) {}
    bool operator<(const ResultStruct &result) const
    {
        return (distance < result.distance);
    }
};

/**
 * Read the database and precompute the statistic for reuse
 * input : csvFilePath of the objectDB
 * output: vector<ObjectStruct> &objectLists of all object extracted from csvFilePath
 * output: StatStruct &commonStat for the standard deviation squares and average distance to be used by all
 */
int computeDB(string &csvFilePath, vector<ObjectStruct> &objectLists, StatStruct &commonStat);

/**
 * Read the database and find the closest object based on normalize euclidean distance
 * input : vector of computed features
 * input : vector<ObjectStruct> &objectLists
 * input : StatStruct &commonStat for the standard deviation squares and average distance to be used by all
 * Output: result name of the closest object
 * input : value of k, default is 1, which become finding the closest neighbour
 */
int classifyObject(vector<double> &features, vector<ObjectStruct> &objectLists, StatStruct &commonStat, string &result, int k = 1);

#endif