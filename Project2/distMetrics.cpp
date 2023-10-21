/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/06/2023
  Description   : All distMetrics to compute the distance between two vectors
*/
#include "distMetrics.h"

/**
 * Calculate the sum of square difference between two feature vectors
 */
int sum_of_squared_difference(vector<int> &targetFeature, vector<int> &srcFeature, float &result)
{
    // ensure both feature vectors are of equal size
    if (targetFeature.size() != srcFeature.size())
    {
        return -1;
    }

    // Loop through each value pair, compute and update the sumOfSquare
    float sumOfSquare = 0;
    for (int i = 0; i < targetFeature.size(); i++)
    {
        sumOfSquare += pow((targetFeature[i] - srcFeature[i]), 2.0);
    }
    result = sumOfSquare;
    return 0;
}

/**
 * Calculate the sum of square difference between two feature vectors (of float numbers)
 */
int sum_of_squared_difference(vector<float> &targetFeature, vector<float> &srcFeature, float &result)
{
    // ensure both feature vectors are of equal size
    if (targetFeature.size() != srcFeature.size())
    {
        return -1;
    }

    // Loop through each value pair, compute and update the sumOfSquare
    float sumOfSquare = 0;
    for (int i = 0; i < targetFeature.size(); i++)
    {
        sumOfSquare += pow((targetFeature[i] - srcFeature[i]), 2.0);
    }
    result = sumOfSquare;
    return 0;
}

/**
 * Calculate the histogram intersection between two histogram
 */
int histogram_intersect(vector<float> &targetFeature, vector<float> &srcFeature, float &result)
{
    // ensure both feature vectors are of equal size
    if (targetFeature.size() != srcFeature.size())
    {
        return -1;
    }
    // Loop through each value pair, compute and update the intersection total
    float interTotal = 0;
    for (int i = 0; i < targetFeature.size(); i++)
    {
        interTotal += std::min(targetFeature[i], srcFeature[i]);
    }

    // The distance between two vectors are 1 - sum of intersection
    result = 1 - interTotal;
    return 0;
}

/**
 * Calculate the distance between two vectors
 * Using distOption chosen - 1 for sum_of_squared, 2 - for histogram intersection
 * store the difference in result
 */
int getDistance(vector<float> &targetFeature, vector<float> &srcFeature, float &result, const int distOption)
{
    switch (distOption)
    {
    case 1:
    {
        if (sum_of_squared_difference(targetFeature, srcFeature, result) != 0)
        {
            return (-1);
        }
        break;
    }
    case 2:
    {
        if (histogram_intersect(targetFeature, srcFeature, result) != 0)
        {
            return (-1);
        }
        break;
    }
    default:
        break;
    }
    return (0);
}