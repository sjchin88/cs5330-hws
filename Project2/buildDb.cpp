/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/06/2023
  Description   : Build the feature vectors database
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
#include "utils.h"
#include "featureCalcs.h"
#include "csv_util.h"

using namespace std;
using namespace cv;

/**
 * Helper function to compute the feature vectors
 * based on imageList
 * and append it to save csv in outputName
 * according to selectedIdx for the feature computation method
 * and zoom factor if you only focus on part of the image
 */
int computeNSave(vector<string> &imageList, char *outputName, int selectedIdx, float zoomFactor);

/*
  Given a directory on the command line, scans through the directory for image files.
  Compute the feature vectors based on chosen method,
  and store the feature vectors in an output csv file
 */
int main(int argc, char *argv[])
{
    // check for sufficient arguments
    // argv[1] = image database, argv[2] = defaultSavePath, argv[3] = feature option, argv[4] = zoom factor
    // zoom factor only required if you choose method 5 & 7
    if (argc < 4)
    {
        printf("usage: %s <image directory path> <csv output directory> <feature option>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    // Getting the variables from command line

    // parse for image database directory from argv[1]
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

    // Parse the output directory for save file from argv[2]
    char outputName[256];
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

    // Parse the feature option from argv[3]
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

    // Parse the zoom factor if selectedIdx is 5 or 7 from argv[4]
    float zoomFactor = 1.0;
    if (selectedIdx == 5 || selectedIdx == 7)
    {
        if (argc < 5)
        {
            printf("missing zoom factor");
            printf("usage: %s <image directory path> <csv output directory> <feature option> <zoom factor>\n", argv[0]);
            exit(EXIT_FAILURE);
        }
        try
        {
            zoomFactor = stof(argv[4]);
        }
        catch (std::exception)
        {
            std::cout << "Error parsing zoom factor " << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Update outputName for csv file based on chosen method option
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
    // For part 5, zoom img with whole texture based on gradient magnitude image
    case 5:
        saveFileName = saveDir + "zoomColorNTextHist.csv";
        break;
    // For part 5, img with whole texture based on gabor filtered image
    case 6:
        saveFileName = saveDir + "ColorNGaborHist.csv";
        break;
    // For part 5, zoom img with whole texture based on gabor filtered image
    case 7:
        saveFileName = saveDir + "zoomColorNGaborHist.csv";
        break;
    default:
        break;
    }
    strcpy(outputName, saveFileName.c_str());

    // call the util function to read in all image file names
    vector<string> imgList;
    if (readImgFiles(dirname, imgList) != 0)
    {
        exit(EXIT_FAILURE);
    };
    // pass to helper method
    if (computeNSave(imgList, outputName, selectedIdx, zoomFactor) != 0)
    {
        exit(EXIT_FAILURE);
    };

    exit(EXIT_SUCCESS);
}

/**
 * Helper function to compute the feature vectors
 * based on imageList
 * and append it to save csv in outputName
 * according to selectedIdx for the feature computation method
 * and zoom factor if you only focus on part of the image
 */
int computeNSave(vector<string> &imageList, char *outputName, int selectedIdx, float zoomFactor)
{
    // only selectedIdx == 1 use int feature vectors
    // the rest are float
    bool isFloat = selectedIdx == 1 ? false : true;

    // Loop trough each image and compute the feature
    for (string fname : imageList)
    {
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

        char imgFname[256];
        strcpy(imgFname, fname.c_str());
        vector<int> featuresVecInt;
        vector<float> featuresVecFloat;

        switch (selectedIdx)
        {
        case 1:
        {
            if (calcBaseline(img, featuresVecInt) != 0)
            {
                cout << "Error in computing feature vectors for" << imgFname << endl;
                break;
            }

            break;
        }
        case 2:
        {
            if (calcRGHist(img, featuresVecFloat, 16) != 0)
            {
                cout << "Error in computing feature vectors for" << imgFname << endl;
                break;
            }
            break;
        }
        case 3:
        {
            if (calcMultiHistLR(img, featuresVecFloat, 16) != 0)
            {
                cout << "Error in computing feature vectors for" << imgFname << endl;
                break;
            }
            break;
        }
        case 4:
        {
            if (calcRGBNTexture(img, featuresVecFloat) != 0)
            {
                cout << "Error in computing feature vectors for" << imgFname << endl;
                break;
            }
            break;
        }
        case 5:
        {
            if (calcRGBNTexture(img, featuresVecFloat, zoomFactor) != 0)
            {
                cout << "Error in computing feature vectors for" << imgFname << endl;
                break;
            }
            break;
        }
        case 6:
        {
            if (calcRGBNGabor(img, featuresVecFloat) != 0)
            {
                cout << "Error in computing feature vectors for" << imgFname << endl;
                break;
            }
            break;
        }
        case 7:
        {
            // auto start = std::chrono::system_clock::now();
            if (calcRGBNGabor(img, featuresVecFloat, zoomFactor) != 0)
            {
                cout << "Error in computing feature vectors for" << imgFname << endl;
                break;
            }
            // Debug timing code
            /* auto end = std::chrono::system_clock::now();
             std::chrono::duration<double> elapsed_seconds = end - start;
             std::cout << "elapsed time for image " << fname << ":" << elapsed_seconds.count() << "s"
                      << std::endl; */

            break;
        }
        default:
            break;
        }
        if (isFloat)
        {
            if (append_image_data_csv(outputName, imgFname, featuresVecFloat, 0) != 0)
            {
                cout << "Error in writing feature vectors to csv for " << imgFname << endl;
            }
        }
        else
        {
            if (append_image_data_csv(outputName, imgFname, featuresVecInt, 0) != 0)
            {
                cout << "Error in writing feature vectors to csv for " << imgFname << endl;
            }
        }
    }
    return 0;
}