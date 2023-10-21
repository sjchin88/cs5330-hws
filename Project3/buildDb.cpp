/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   : Build the feature vectors database for labeled object
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdlib.h>
#include <dirent.h>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <ctime>
#include "features.h"
#include "file_util.h"

using namespace std;
using namespace cv;

/**
 * Helper function to compute the feature vectors based on imageList
 * and append it to save csv in saveFileName
 * input : imageList , list of full path for the image
 * input : saveFileName, location to save the csv file
 * input : objNames, the obj name to be used for each image
 */
int computeNSave(vector<string> &imageList, char *saveFileName, vector<string> &objNames);

/*
  Given a directory on the command line, scans through the directory for image files.
  Compute the feature vectors
  and store the feature vectors in an output csv file
 */
int main(int argc, char *argv[])
{
    // check for sufficient arguments
    // argv[1] = training set directory, argv[2] = defaultSavePath
    if (argc < 3)
    {
        printf("usage: %s <training set directory path> <csv output directory> \n", argv[0]);
        exit(EXIT_FAILURE);
    }
    // Getting the variables from command line

    // parse for training set directory from argv[1]
    char dirname[256];
    try
    {
        strcpy_s(dirname, argv[1]);
    }
    catch (std::exception)
    {
        std::cout << "Invalid directory for training set " << std::endl;
        exit(EXIT_FAILURE);
    }

    // Parse the output directory for save file from argv[2]

    string saveDir;
    try
    {
        saveDir = argv[2];
    }
    catch (std::exception)
    {
        std::cout << "Invalid directory for output csv file" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Update saveFileName for csv file based on chosen method option
    string saveFileN = saveDir + "objectDB.csv";
    char saveFileName[256];
    strcpy_s(saveFileName, saveFileN.c_str());

    // call the util function to read in all image file names
    vector<string> imgList;
    vector<string> objNames;
    if (readImgFiles(dirname, imgList, objNames) != 0)
    {
        exit(EXIT_FAILURE);
    };
    // pass to helper method
    if (computeNSave(imgList, saveFileName, objNames) != 0)
    {
        exit(EXIT_FAILURE);
    };

    exit(EXIT_SUCCESS);
}

/**
 * Helper function to compute the feature vectors based on imageList
 * and append it to save csv in saveFileName
 */
int computeNSave(vector<string> &imageList, char *saveFileName, vector<string> &objNames)
{
    // Loop trough each image and compute the feature
    for (int id = 0; id < imageList.size(); id++)
    {
        string fname = imageList[id];
        cv::Mat img;
        try
        {
            img = imread(fname);
        }
        catch (exception)
        {
            cout << "error in reading the image for " << fname << endl;
            continue;
        }

        char objName[256];
        strcpy_s(objName, objNames[id].c_str());

        vector<vector<double>> featureList;
        if (processNCompute(img, featureList) != 0)
        {
            cout << "error in computing features for " << fname << endl;
            continue;
        }
        // see if there is a waiting keystroke
        char key = cv::waitKey(0);
        if (featureList.size() > 1)
        {
            cout << "more than one object presents for img " << fname << endl;
            continue;
        }

        if (featureList.size() < 1)
        {
            cout << "no features extracted for " << fname << endl;
            continue;
        }

        // Reconstruct the features in float
        vector<float> features;
        for (int i = 0; i < F_FEATURES_SIZE; i++)
        {
            features.push_back(featureList[0][i]);
        }

        if (append_image_data_csv(saveFileName, objName, features, 0) != 0)
        {
            cout << "Error in writing feature vectors to csv for " << objName << endl;
        }
    }
    return 0;
}
