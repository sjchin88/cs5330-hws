/*
  CSJ
  Build the feature vectors database
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
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
#include "distMetrics.h"

using namespace std;
using namespace cv;

struct ResultStruct
{
  char *imgName;
  float distance;
  ResultStruct(char *name, float dist) : imgName(name), distance(dist) {}
  bool operator<(const ResultStruct &result) const
  {
    return (distance < result.distance);
  }
};

int searchBaseline(cv::Mat &targetImg, string &csvFile, vector<ResultStruct> &resultList);
int searchRGHist(cv::Mat &targetImg, string &csvFile, const int histSize, vector<ResultStruct> &resultList);
int searchMultiHistLR(cv::Mat &targetImg, string &csvFile, const int histSize, vector<ResultStruct> &resultList);
int searchRGBNTexture(cv::Mat &targetImg, string &csvFilePath, vector<ResultStruct> &resultList);

/*
  Given a directory on the command line, scans through the directory for image files.

  Compute the feature vectors based on chosen method,
  and store the feature vectors in an output csv file
 */
int main(int argc, char *argv[])
{
  // check for sufficient arguments
  if (argc < 5)
  {
    printf("usage: %s <target image path> <csv output directory> <feature option> <N (number of closest images to be displayed)>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  // Getting the variables from command line
  string targetImagePath;
  cv::Mat targetImage;
  try
  {
    targetImagePath = argv[1];
    targetImage = imread(targetImagePath);
  }
  catch (std::exception)
  {
    std::cout << "Unable to read target image " << std::endl;
    exit(EXIT_FAILURE);
  }

  string csvDir;
  try
  {
    csvDir = argv[2];
  }
  catch (std::exception)
  {
    std::cout << "Invalid csv files directory " << std::endl;
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

  // Parse the N
  int topN;
  try
  {
    topN = stoi(argv[4]);
  }
  catch (std::exception)
  {
    std::cout << "Error parsing selected idx " << std::endl;
    exit(EXIT_FAILURE);
  }

  vector<ResultStruct> resultList;

  switch (selectedIdx)
  {
  case 1:
  {
    string csvFilePath = csvDir + "baseline.csv";
    if (searchBaseline(targetImage, csvFilePath, resultList) != 0)
    {
      std::cout << "Error in searching database" << endl;
      exit(EXIT_FAILURE);
    };
    break;
  }
  case 2:
  {
    string csvFilePath = csvDir + "rghistogram.csv";
    if (searchRGHist(targetImage, csvFilePath, 16, resultList) != 0)
    {
      std::cout << "Error in searching database" << endl;
      exit(EXIT_FAILURE);
    };
    break;
  }
  case 3:
  {
    string csvFilePath = csvDir + "multihistogram.csv";
    if (searchMultiHistLR(targetImage, csvFilePath, 16, resultList) != 0)
    {
      std::cout << "Error in searching database" << endl;
      exit(EXIT_FAILURE);
    };
    break;
  }
  case 4:
  {
    string csvFilePath = csvDir + "colorNTextureHist.csv";
    if (searchRGBNTexture(targetImage, csvFilePath, resultList) != 0)
    {
      std::cout << "Error in searching database" << endl;
      exit(EXIT_FAILURE);
    };
    break;
  }
  }
  sort(resultList.begin(), resultList.end());
  for (int i = 0; i < topN; i++)
  {
    ResultStruct res = resultList[i];
    std::cout << res.imgName << " " << res.distance << endl;
  }
}

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

int searchMultiHistLR(cv::Mat &targetImg, string &csvFilePath, const int histSize, vector<ResultStruct> &resultList)
{
  try
  {
    vector<float> featuresVec;
    calcMultiHistLR(targetImg, featuresVec, histSize);
    int halfCnt = histSize * histSize;
    vector<float> leftFeatures = vector<float>(featuresVec.begin(), featuresVec.begin() + halfCnt);
    vector<float> rightFeatures = vector<float>(featuresVec.begin() + halfCnt, featuresVec.end());
    vector<char *> imgNames;
    vector<vector<float>> imgData;
    char csvPath[256];
    strcpy(csvPath, csvFilePath.c_str());
    read_image_data_csv(csvPath, imgNames, imgData);
    int size = imgNames.size();
    for (int i = 0; i < size; i++)
    {
      float diffLeft;
      float diffRight;
      vector<float> dataLeft = vector<float>(imgData[i].begin(), imgData[i].begin() + halfCnt);
      vector<float> dataRight = vector<float>(imgData[i].begin() + halfCnt, imgData[i].end());
      histogram_intersect(leftFeatures, dataLeft, diffLeft);
      histogram_intersect(rightFeatures, dataRight, diffRight);
      resultList.push_back(ResultStruct(imgNames[i], 0.5 * diffLeft + 0.5 * diffRight));
    }
  }
  catch (exception)
  {
    return (-1);
  }

  return 0;
}

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
    vector<float> leftFeatures = vector<float>(featuresVec.begin(), featuresVec.begin() + halfCnt);
    vector<float> rightFeatures = vector<float>(featuresVec.begin() + halfCnt, featuresVec.end());
    vector<char *> imgNames;
    vector<vector<float>> imgData;
    char csvPath[256];
    strcpy(csvPath, csvFilePath.c_str());
    read_image_data_csv(csvPath, imgNames, imgData);
    int size = imgNames.size();
    for (int i = 0; i < size; i++)
    {
      float diffRGB;
      float diffTexture;
      vector<float> dataRGB = vector<float>(imgData[i].begin(), imgData[i].begin() + halfCnt);
      vector<float> dataTexture = vector<float>(imgData[i].begin() + halfCnt, imgData[i].end());
      histogram_intersect(leftFeatures, dataRGB, diffRGB);
      histogram_intersect(rightFeatures, dataTexture, diffTexture);
      resultList.push_back(ResultStruct(imgNames[i], 0.5 * diffRGB + 0.5 * diffTexture));
    }
  }
  catch (exception)
  {
    return (-1);
  }

  return 0;
}