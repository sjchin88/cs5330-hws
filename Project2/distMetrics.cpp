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