/*
  CSJ
  Build the feature vectors database
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
#include "utils.h"
#include "featureCalcs.h"
#include "csv_util.h"

using namespace std;
using namespace cv;

/**
 * Helper function to compute the feature vectors and append it to save csv
 * according to selectedIdx
 */
int computeNSave(vector<string> &imageList, char *outputName, int selectedIdx);

/*
  Given a directory on the command line, scans through the directory for image files.

  Compute the feature vectors based on chosen method,
  and store the feature vectors in an output csv file
 */
int main(int argc, char *argv[])
{
    // check for sufficient arguments
    if (argc < 4)
    {
        printf("usage: %s <image directory path> <csv output directory> <feature option>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    // Getting the variables from command line
    // argv[1] = image database, argv[2] = defaultSavePath, argv[3] = feature option
    char dirname[256];
    try
    {
        strcpy(dirname, argv[1]);
    }
    catch (std::exception)
    {
        std::cout << "Invalid directory for image database " << std::endl;
        exit(EXIT_FAILURE);
    }

    // Parse the output directory for save file
    char outputName[256];
    string saveDir;
    // get the directory path
    // string saveDir = "C:/CS5330_Assets/olympus/";
    // strcpy(dirname, saveDir.c_str());
    try
    {
        saveDir = argv[2];
    }
    catch (std::exception)
    {
        std::cout << "Invalid directory for output csv file" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Parse the option
    int selectedIdx;
    try
    {
        selectedIdx = stoi(argv[3]);
    }
    catch (std::exception)
    {
        std::cout << "Error parsing selected idx " << std::endl;
        exit(EXIT_FAILURE);
    }

    /*
     * Default selection for feature computation is baseline 'b'
     * Other valid selections include 'h' for Histogram Matching
     * 'm' for multi-histogram matching
     * 't' for texture matching
     */
    // char selection;

    string saveFileName;
    switch (selectedIdx)
    {
    case 1:
        saveFileName = saveDir + "baseline.csv";
        break;
    case 2:
        saveFileName = saveDir + "rghistogram.csv";
        break;
    case 3:
        saveFileName = saveDir + "multihistogram.csv";
        break;
    case 4:
        saveFileName = saveDir + "colorNTextureHist.csv";
        break;
    default:
        break;
    }

    strcpy(outputName, saveFileName.c_str());

    // call the util function to read in all image file names
    vector<string> imgList;
    readImgFiles(dirname, imgList);
    computeNSave(imgList, outputName, selectedIdx);

    exit(EXIT_SUCCESS);
}

int computeNSave(vector<string> &imageList, char *outputName, int selectedIdx)
{
    // Loop trough each image and compute the feature
    for (string fname : imageList)
    {
        cv::Mat img = imread(fname);
        char imgFname[256];
        strcpy(imgFname, fname.c_str());

        switch (selectedIdx)
        {
        case 1:
        {
            vector<int> featuresVec;
            if (calcBaseline(img, featuresVec) != 0)
            {
                cout << "Error in computing feature vectors for" << imgFname << endl;
                break;
            }
            if (append_image_data_csv(outputName, imgFname, featuresVec, 0) != 0)
            {
                cout << "Error in writing feature vectors to csv for " << imgFname << endl;
                break;
            }
            break;
        }
        case 2:
        {
            vector<float> featuresVec;
            if (calcRGHist(img, featuresVec, 16) != 0)
            {
                cout << "Error in computing feature vectors for" << imgFname << endl;
                break;
            }
            if (append_image_data_csv(outputName, imgFname, featuresVec, 0) != 0)
            {
                cout << "Error in writing feature vectors to csv for " << imgFname << endl;
                break;
            }
            break;
        }
        case 3:
        {
            vector<float> featuresVec;
            if (calcMultiHistLR(img, featuresVec, 16) != 0)
            {
                cout << "Error in computing feature vectors for" << imgFname << endl;
                break;
            }
            if (append_image_data_csv(outputName, imgFname, featuresVec, 0) != 0)
            {
                cout << "Error in writing feature vectors to csv for " << imgFname << endl;
                break;
            }
            break;
        }
        case 4:
        {
            vector<float> featuresVec;
            if (calcRGBNTexture(img, featuresVec) != 0)
            {
                cout << "Error in computing feature vectors for" << imgFname << endl;
                break;
            }
            if (append_image_data_csv(outputName, imgFname, featuresVec, 0) != 0)
            {
                cout << "Error in writing feature vectors to csv for " << imgFname << endl;
                break;
            }
            break;
        }
        }
    }
    return 0;
}