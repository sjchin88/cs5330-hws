/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   : 2-D object recognition for real time video feed
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

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    // check for sufficient arguments
    if (argc < 4)
    {
        printf("usage: %s <cameraIdx> <defaultSavePath> <objectDb csv file path>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Getting the variables from command line
    // argv[1] = cameraIdx, argv[2] = defaultSavePath, argv[3] = objectDb csv file path
    // Set the cameraIdx from argv[1], and set the videocapture element
    VideoCapture *capdev;
    int cameraIdx;
    try
    {
        cameraIdx = std::stoi(argv[1]);
    }
    catch (std::exception)
    {
        // Set it to 0
        cameraIdx = 0;
    }
    cout << "camera idx is " << cameraIdx << endl;

    // open the video device
    capdev = new cv::VideoCapture(cameraIdx);
    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        exit(EXIT_FAILURE);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);
    int fps = (int)capdev->get(cv::CAP_PROP_FPS);
    printf("Captured fps: %d \n", fps);

    // Set default fps for saving
    int savingFPS = 10;
    if (fps > 0)
    {
        savingFPS = fps;
    }

    // Set default save path from argv[2]
    String defaultSavePath;
    try
    {
        defaultSavePath = argv[2];
    }
    catch (std::exception)
    {
        std::cout << "Could not read the default save path " << std::endl;
        exit(EXIT_FAILURE);
    }
    cout << "save path is " << defaultSavePath << endl;

    // Parse the objectDB csv file path from argv[3]
    string csvFilePath;
    try
    {
        csvFilePath = argv[3];
    }
    catch (std::exception)
    {
        std::cout << "Invalid csv files directory " << std::endl;
        exit(EXIT_FAILURE);
    }

    int topN = 1;
    // Parse the topN number from argv[3] if present
    if (argc >= 5)
    {
        try
        {
            topN = stoi(argv[4]);
        }
        catch (std::exception)
        {
            printf("Invalid argument for topN number, was %s . Default will be used", argv[3]);
            topN = 1;
        }
    }

    int kValue = 1;
    // Parse the KNN k-value number from argv[3] if present
    if (argc >= 6)
    {
        try
        {
            kValue = stoi(argv[5]);
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

    // Initialize variables required
    namedWindow("Video", 1); // identifies a window
    Mat capturedFrame;
    Mat tempFrame;
    Mat displayFrame;
    char selection = 'R';
    // Initialize all boolean flags for display option
    bool showThreshold = false;
    bool showCC = false;
    bool showAR = false;
    bool showFeatures = false;
    bool videoSaving = false;

    cv::VideoWriter videoWriter;

    // Run the while loop
    while (true)
    {
        *capdev >> capturedFrame; // get a new frame from the camera, treat as a stream

        if (capturedFrame.empty())
        {
            printf("frame is empty\n");
            break;
        }

        // Default is to set the displayFrame as capturedFrame
        displayFrame = capturedFrame;

        vector<vector<double>> features;
        processNCompute(capturedFrame, features, topN, false, showThreshold, false, showCC, showAR);
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
            cv::putText(displayFrame, result, center, cv::FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255));
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
                        cv::putText(displayFrame, featureText[j], center, cv::FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255));
                    }
                }
            }
        }

        // check if video saving is on
        if (videoSaving)
        {
            videoWriter.write(displayFrame);
        }

        cv::imshow("Video", displayFrame);

        // see if there is a waiting keystroke every 33ms
        char key = cv::waitKey(33);

        switch (key)
        {

        case 't':
            showThreshold = !showThreshold;
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

        //  keystroke v turn on saving video to the file
        case 'v':
        {
            // if videoSaving is on, turn it off
            if (videoSaving)
            {
                videoWriter.release();
                videoSaving = false;
            }
            else
            {
                // Set the video writer
                std::time_t timeStamp = std::time(nullptr);
                String videoPath = defaultSavePath + "video_" + to_string(timeStamp) + ".avi";
                /*  cv::VideoWriter::VideoWriter (const String &filename, int fourcc, double fps,Size frameSize,bool isColor = true
                        )	 */
                videoWriter = cv::VideoWriter(videoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), savingFPS, refS);
                videoSaving = true;
            }

            break;
        }

        default:
            break;
        }

        // Task 2: keystroke s save the current display frame
        if (key == 's')
        {
            std::time_t timeStamp = std::time(nullptr);
            String finalPath = defaultSavePath + "screenshot_" + to_string(timeStamp) + ".png";
            imwrite(finalPath, displayFrame);
        }

        // keystroke q quit the program
        if (key == 'q')
        {
            break;
        }
    }

    delete capdev;
    return (0);
}