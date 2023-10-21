/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   : imgDisplay program from Project 1, used to verify the effect of all implemented functions
                and also classify new image, save new label if necessary
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cvdef.h>
#include <vector>
#include <iostream>
#include "filters.h"
#include "file_util.h"
#include "features.h"
#include "classify.h"

using namespace std;
using namespace cv;

/**
 * Main function
 */
int main(int argc, char *argv[])
{
    // check for sufficient arguments
    if (argc < 3)
    {
        printf("usage: %s <directory of target images> <objectDb csv file path> <top N number of connected region to be identified> <k-value for K-Nearest Neighbor matching>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Getting the variables from command line
    // argv[1] = directory of target images, argv[2] = objectDb csv file path,
    // parse for directory of target images from argv[1]
    char dirname[256];
    try
    {
        strcpy_s(dirname, argv[1]);
    }
    catch (std::exception)
    {
        std::cout << "Invalid directory for target image " << std::endl;
        exit(EXIT_FAILURE);
    }

    // Parse the objectDB csv file path from argv[2]
    string csvFilePath;
    try
    {
        csvFilePath = argv[2];
    }
    catch (std::exception)
    {
        std::cout << "Invalid csv files directory " << std::endl;
        exit(EXIT_FAILURE);
    }

    int topN = 1;
    // Parse the topN number from argv[3] if present
    if (argc >= 4)
    {
        try
        {
            topN = stoi(argv[3]);
        }
        catch (std::exception)
        {
            printf("Invalid argument for topN number, was %s . Default will be used", argv[3]);
            topN = 1;
        }
    }

    int kValue = 1;
    // Parse the KNN k-value number from argv[3] if present
    if (argc >= 5)
    {
        try
        {
            kValue = stoi(argv[4]);
        }
        catch (std::exception)
        {
            printf("Invalid argument for topN number, was %s . Default will be used", argv[4]);
            kValue = 1;
        }
    }

    // Precompute the common statistic
    vector<ObjectStruct> objectLists;
    double tempNum = 0;
    vector<double> tempStd;
    StatStruct commonStat(tempStd, tempNum);
    computeDB(csvFilePath, objectLists, commonStat);

    // Initialize all boolean flags for display option
    bool showInterim = false;
    bool showThreshold = false;
    bool showMorpho = false;
    bool showCC = false;
    bool showAR = false;
    bool showFeatures = false;

    // call the util function to read in all image file names
    vector<string> imgList;
    vector<string> objNames;
    if (readImgFiles(dirname, imgList, objNames) != 0)
    {
        exit(EXIT_FAILURE);
    };

    // Loop trough each image and process the image
    for (int id = 0; id < imgList.size(); id++)
    {
        string fname = imgList[id];
        cv::Mat img;
        try
        {
            img = imread(fname);
            if (img.empty())
            {
                std::cout << "Could not read the image: " << fname << std::endl;
                continue;
            }
        }
        catch (exception)
        {
            cout << "error in reading the image for " << fname << endl;
            continue;
        }

        // Initialize variables required
        Mat displayImg;
        Mat tempImg;
        vector<vector<double>> features;

        // Run the while loop
        while (true)
        {
            displayImg = img.clone();
            processNCompute(img, features, topN, showInterim, showThreshold, showMorpho, showCC, showAR);
            for (int i = 0; i < features.size(); i++)
            {
                string result;
                classifyObject(features[i], objectLists, commonStat, result, kValue);

                // Putting the text
                int fontFace = FONT_HERSHEY_SIMPLEX;
                int thickness = 3;
                int baseline = 0;
                Size textSize = getTextSize(result, fontFace, 1.5, thickness, &baseline);
                baseline += thickness;

                cv::Point2i center(features[i][F_CENTROID_X] - textSize.width / 2, features[i][F_CENTROID_Y] - textSize.height);
                cv::putText(displayImg, result, center, cv::FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0));
                if (showFeatures)
                {
                    vector<string> featureText;
                    if (getFeatureText(features[i], featureText) == 0)
                    {
                        for (int j = 0; j < featureText.size(); j++)
                        {
                            textSize = getTextSize(featureText[j], fontFace, 0.7, thickness, &baseline);
                            center.x = features[i][F_CENTROID_X] - textSize.width / 2;
                            center.y = features[i][F_CENTROID_Y] + textSize.height * j;
                            cv::putText(displayImg, featureText[j], center, cv::FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0));
                        }
                    }
                }
            }

            // Show the display image
            imshow("Current Image", displayImg);

            // see if there is a waiting keystroke
            char key = cv::waitKey(0);
            switch (key)
            {
            case 'i':
                showInterim = !showInterim;
                break;
            case 't':
                showThreshold = !showThreshold;
                break;
            case 'm':
                showMorpho = !showMorpho;
                break;
            case 'c':
                showCC = !showCC;
                break;
            case 'z':
                showAR = !showAR;
                break;
            case 'x':
                showFeatures = !showFeatures;
                break;
            }

            // Keystroke n prompt for user input for new label
            if (key == 'n')
            {
                char label[256];
                cout << "please input the label for current object to be put into db" << endl;
                cin >> label;
                char outputName[256];
                strcpy_s(outputName, csvFilePath.c_str());
                // Reconstruct the features in float
                vector<float> newObjFeatures;
                for (int i = 0; i < F_FEATURES_SIZE; i++)
                {
                    newObjFeatures.push_back(features[0][i]);
                }
                append_image_data_csv(outputName, label, newObjFeatures);
            }

            // Keystroke a go to next images
            if (key == 'a')
            {
                break;
            }

            // The program will quit if the user press 'q'
            if (key == 'q')
            {
                exit(EXIT_SUCCESS);
            }
        }
    }

    exit(EXIT_SUCCESS);
}