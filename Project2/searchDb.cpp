/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/06/2023
  Description   : All required method to search the database
*/
#include "searchDb.h"

/**
 * compute the distance in baseline database located in csvFilePath
 * compare to the targetImg
 * store all the result in resultList
 * For baseline, default is using sum of square distance method
 */
int searchBaseline(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList)
{
    try
    {
        vector<int> featuresVec;
        if (calcBaseline(targetImg, featuresVec) != 0)
        {
            cout << "error computing feature vectors for target image" << endl;
            return (-1);
        };
        vector<char *> imgNames;
        vector<vector<int>> imgData;
        char csvPath[256];
        strcpy(csvPath, csvFilePath.c_str());

        if (read_image_data_csv(csvPath, imgNames, imgData) != 0)
        {
            cout << "error reading image database csv file " << endl;
            return (-1);
        };
        int size = imgNames.size();
        for (int i = 0; i < size; i++)
        {
            float diff;
            // Push to resultList if calculation successful
            if (sum_of_squared_difference(featuresVec, imgData[i], diff) == 0)
            {
                resultList.push_back(ResultStruct(imgNames[i], diff));
            };
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return 0;
}

/**
 * compute the distance in rg histogram database located in csvFilePath
 * compare to the targetImg
 * store all the result in resultList
 * histSize determine the size of n-bins used
 * distIdx determine which dist metrics to use
 * 1 - for sum of square, 2 - for histogram intersection
 */
int searchRGHist(cv::Mat &targetImg, string &csvFilePath, const int histSize, vector<ResultStruct> &resultList, const int distIdx)
{
    try
    {
        vector<float> featuresVec;
        if (calcRGHist(targetImg, featuresVec, histSize) != 0)
        {
            cout << "error computing feature vectors for target image" << endl;
            return (-1);
        };
        vector<char *> imgNames;
        vector<vector<float>> imgData;
        char csvPath[256];
        strcpy(csvPath, csvFilePath.c_str());
        if (read_image_data_csv(csvPath, imgNames, imgData) != 0)
        {
            cout << "error reading image database csv file " << endl;
            return (-1);
        };

        int size = imgNames.size();
        for (int i = 0; i < size; i++)
        {
            float diff;
            if (getDistance(featuresVec, imgData[i], diff, distIdx) == 0)
            {
                resultList.push_back(ResultStruct(imgNames[i], diff));
            };
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return 0;
}

/**
 * Helper function to compute the distance for feature vectors based on
 * two different features combined together
 * halfCnt determine the separation between the first and second feature vector
 * firstWeight & secWeight determine the respective weight of the distance metrics
 */
int twoHistIntersect(vector<float> &featuresVec, const int halfCnt, string &csvFilePath, const float firstWeight, const float secWeight, vector<ResultStruct> &resultList, const int distIdx)
{
    try
    {
        // Separate target features into two half
        vector<float> leftFeatures = vector<float>(featuresVec.begin(), featuresVec.begin() + halfCnt);
        vector<float> rightFeatures = vector<float>(featuresVec.begin() + halfCnt, featuresVec.end());

        // read the image_data
        vector<char *> imgNames;
        vector<vector<float>> imgData;
        char csvPath[256];
        strcpy(csvPath, csvFilePath.c_str());
        if (read_image_data_csv(csvPath, imgNames, imgData) != 0)
        {
            cout << "error reading image database csv file " << endl;
            return (-1);
        };

        // Loop through each image and computed feature vectors
        int size = imgNames.size();
        for (int i = 0; i < size; i++)
        {
            // Separate the feature vectors into half
            vector<float> dataLeft = vector<float>(imgData[i].begin(), imgData[i].begin() + halfCnt);
            vector<float> dataRight = vector<float>(imgData[i].begin() + halfCnt, imgData[i].end());

            // Compute the histogram intersection distance value separately
            float diffLeft;
            float diffRight;
            if (getDistance(leftFeatures, dataLeft, diffLeft, distIdx) == 0 &&
                getDistance(rightFeatures, dataRight, diffRight, distIdx) == 0)
            {
                // Combine the distance according to predetermined weights
                resultList.push_back(ResultStruct(imgNames[i], firstWeight * diffLeft + secWeight * diffRight));
            }
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return 0;
}

/**
 * compute the distance in multiHistogram (with Left & Right part) database located in csvFilePath
 * compare to the targetImg
 * store all the result in resultList
 * histSize determine the size of n-bins used
 * distIdx determine which dist metrics to use
 * 1 - for sum of square, 2 - for histogram intersection
 */
int searchMultiHistLR(cv::Mat &targetImg, string &csvFilePath, const int histSize, vector<ResultStruct> &resultList, const int distIdx)
{
    try
    {
        vector<float> featuresVec;
        if (calcMultiHistLR(targetImg, featuresVec, histSize) != 0)
        {
            cout << "error computing feature vectors for target image" << endl;
            return (-1);
        };
        int halfCnt = histSize * histSize;
        if (twoHistIntersect(featuresVec, halfCnt, csvFilePath, 0.5, 0.5, resultList, distIdx) != 0)
        {
            return (-1);
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return 0;
}

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
int searchRGBNTexture(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList, const int distIdx, float zoomFactor)
{
    try
    {
        vector<float> featuresVec;
        if (calcRGBNTexture(targetImg, featuresVec, zoomFactor) != 0)
        {
            return (-1);
        }
        int halfCnt = 8 * 8 * 8;
        if (twoHistIntersect(featuresVec, halfCnt, csvFilePath, 0.5, 0.5, resultList, distIdx) != 0)
        {
            return (-1);
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return 0;
}

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
int searchRGBNGabor(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList, const int distIdx, float zoomFactor)
{
    try
    {
        vector<float> featuresVec;
        if (calcRGBNGabor(targetImg, featuresVec, zoomFactor) != 0)
        {
            return (-1);
        }
        int halfCnt = 8 * 8 * 8;
        if (twoHistIntersect(featuresVec, halfCnt, csvFilePath, 0.5, 0.5, resultList, distIdx) != 0)
        {
            return (-1);
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return 0;
}
