#include "distMetrics.h"

/**
 * Calculate the sum of square difference between two feature vectors
 */
int sum_of_squared_difference(vector<int> &targetFeature, vector<int> &srcFeature, float &result)
{
    if (targetFeature.size() != srcFeature.size())
    {
        return -1;
    }
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
    if (targetFeature.size() != srcFeature.size())
    {
        return -1;
    }
    float interTotal = 0;
    for (int i = 0; i < targetFeature.size(); i++)
    {
        interTotal += std::min(targetFeature[i], srcFeature[i]);
    }
    result = 1 - interTotal;
    return 0;
}