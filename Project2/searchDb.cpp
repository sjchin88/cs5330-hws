#include "searchDb.h"

/**
 * compute the distance in baseline database located in csvFilePath
 * compare to the targetImg
 * store all the result in resultList
 */
int searchBaseline(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList)
{
    try
    {
        vector<int> featuresVec;
        calcBaseline(targetImg, featuresVec);
        vector<char *> imgNames;
        vector<vector<int>> imgData;
        char csvPath[256];
        strcpy(csvPath, csvFilePath.c_str());

        read_image_data_csv(csvPath, imgNames, imgData);
        int size = imgNames.size();
        for (int i = 0; i < size; i++)
        {
            float diff;
            sum_of_squared_difference(featuresVec, imgData[i], diff);
            resultList.push_back(ResultStruct(imgNames[i], diff));
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
 */
int searchRGHist(cv::Mat &targetImg, string &csvFilePath, const int histSize, vector<ResultStruct> &resultList)
{
    try
    {
        vector<float> featuresVec;
        calcRGHist(targetImg, featuresVec, histSize);
        vector<char *> imgNames;
        vector<vector<float>> imgData;
        char csvPath[256];
        strcpy(csvPath, csvFilePath.c_str());
        read_image_data_csv(csvPath, imgNames, imgData);
        int size = imgNames.size();
        for (int i = 0; i < size; i++)
        {
            float diff;
            histogram_intersect(featuresVec, imgData[i], diff);
            resultList.push_back(ResultStruct(imgNames[i], diff));
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
int twoHistIntersect(vector<float> &featuresVec, const int halfCnt, string &csvFilePath, const float firstWeight, const float secWeight, vector<ResultStruct> &resultList)
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
        read_image_data_csv(csvPath, imgNames, imgData);

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
            histogram_intersect(leftFeatures, dataLeft, diffLeft);
            histogram_intersect(rightFeatures, dataRight, diffRight);

            // Combine the distance according to predetermined weights
            resultList.push_back(ResultStruct(imgNames[i], firstWeight * diffLeft + secWeight * diffRight));
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
 */
int searchMultiHistLR(cv::Mat &targetImg, string &csvFilePath, const int histSize, vector<ResultStruct> &resultList)
{
    try
    {
        vector<float> featuresVec;
        calcMultiHistLR(targetImg, featuresVec, histSize);
        int halfCnt = histSize * histSize;
        if (twoHistIntersect(featuresVec, halfCnt, csvFilePath, 0.5, 0.5, resultList) != 0)
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
 */
int searchRGBNTexture(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList)
{
    try
    {
        vector<float> featuresVec;
        if (calcRGBNTexture(targetImg, featuresVec) != 0)
        {
            return (-1);
        }
        int halfCnt = 8 * 8 * 8;
        if (twoHistIntersect(featuresVec, halfCnt, csvFilePath, 0.5, 0.5, resultList) != 0)
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
 * compare to the targetImg with specified zoomFactor
 * store all the result in resultList
 */
int searchRGBNTexture(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList, float zoomFactor)
{
    try
    {
        vector<float> featuresVec;
        if (calcRGBNTexture(targetImg, featuresVec, zoomFactor) != 0)
        {
            return (-1);
        }

        int halfCnt = 8 * 8 * 8;
        if (twoHistIntersect(featuresVec, halfCnt, csvFilePath, 0.5, 0.5, resultList) != 0)
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
 */
int searchRGBNGabor(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList)
{
    try
    {
        vector<float> featuresVec;
        if (calcRGBNGabor(targetImg, featuresVec, false) != 0)
        {
            return (-1);
        }
        int halfCnt = 8 * 8 * 8;
        if (twoHistIntersect(featuresVec, halfCnt, csvFilePath, 0.5, 0.5, resultList) != 0)
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
 */
int searchRGBNGabor(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList, float zoomFactor)
{
    try
    {
        vector<float> featuresVec;
        if (calcRGBNGabor(targetImg, featuresVec, zoomFactor) != 0)
        {
            return (-1);
        }
        int halfCnt = 8 * 8 * 8;
        if (twoHistIntersect(featuresVec, halfCnt, csvFilePath, 0.5, 0.5, resultList) != 0)
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