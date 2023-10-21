/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/06/2023
  Description   : search the feature vectors database
                  based on target image
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
#include "searchDb.h"

using namespace std;
using namespace cv;

/*
  Given a directory on the command line, scans through the directory for image files.
  Compute the feature vectors based on chosen method,
  extract the feature vectors of image database from the csv file
  compared it with the target image using distance metrics chosen
  output the top N result (may include the target image itself if present in the database)
 */
int main(int argc, char *argv[])
{
  // check for sufficient arguments
  if (argc < 6)
  {
    printf("usage: %s <target image path> <csv output directory> <feature option> <distance metric option> <N (number of closest images to be displayed)>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Getting the variables from command line
  // Parse targetImage from argv[1]
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

  // Parse csv file directory from argv[2]
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

  // Parse the method option from argv[3]
  int selectedIdx;
  try
  {
    selectedIdx = stoi(argv[3]);
  }
  catch (std::exception)
  {
    std::cout << "Error parsing method selection idx " << std::endl;
    exit(EXIT_FAILURE);
  }

  // Parse the distance metric option from argv[4]
  int distIdx;
  try
  {
    distIdx = stoi(argv[4]);
  }
  catch (std::exception)
  {
    std::cout << "Error parsing distance metrics idx " << std::endl;
    exit(EXIT_FAILURE);
  }

  // Parse the N value from argv[5]
  int topN;
  try
  {
    topN = stoi(argv[5]);
  }
  catch (std::exception)
  {
    std::cout << "Error parsing selected idx " << std::endl;
    exit(EXIT_FAILURE);
  }

  // Parse the zoom factor if selectedIdx is 5 or 7 from argv[6]
  float zoomFactor = 1.0F;
  if (selectedIdx == 5 || selectedIdx == 7)
  {
    if (argc < 7)
    {
      printf("missing zoom factor");
      printf("usage: %s <target image path> <csv output directory> <feature option> <distance metric option> <N (number of closest images to be displayed)> <zoom factor>\n", argv[0]);
      exit(EXIT_FAILURE);
    }
    try
    {
      zoomFactor = stof(argv[6]);
    }
    catch (std::exception)
    {
      std::cout << "Error parsing zoom factor " << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // Search the corresponding database and update the distance to
  // resultList
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
    if (searchRGHist(targetImage, csvFilePath, 16, resultList, distIdx) != 0)
    {
      std::cout << "Error in searching database" << endl;
      exit(EXIT_FAILURE);
    };
    break;
  }
  case 3:
  {
    string csvFilePath = csvDir + "multihistogram.csv";
    if (searchMultiHistLR(targetImage, csvFilePath, 16, resultList, distIdx) != 0)
    {
      std::cout << "Error in searching database" << endl;
      exit(EXIT_FAILURE);
    };
    break;
  }
  case 4:
  {
    string csvFilePath = csvDir + "colorNTextureHist.csv";
    if (searchRGBNTexture(targetImage, csvFilePath, resultList, distIdx) != 0)
    {
      std::cout << "Error in searching database" << endl;
      exit(EXIT_FAILURE);
    };
    break;
  }
  case 5:
  {
    string csvFilePath = csvDir + "zoomColorNTextHist.csv";
    if (searchRGBNTexture(targetImage, csvFilePath, resultList, distIdx, zoomFactor) != 0)
    {
      std::cout << "Error in searching database" << endl;
      exit(EXIT_FAILURE);
    };
    break;
  }
  case 6:
  {
    string csvFilePath = csvDir + "ColorNGaborHist.csv";
    if (searchRGBNGabor(targetImage, csvFilePath, resultList, distIdx) != 0)
    {
      std::cout << "Error in searching database" << endl;
      exit(EXIT_FAILURE);
    };
    break;
  }
  case 7:
  {
    string csvFilePath = csvDir + "zoomColorNGaborHist.csv";
    if (searchRGBNGabor(targetImage, csvFilePath, resultList, distIdx, zoomFactor) != 0)
    {
      std::cout << "Error in searching database" << endl;
      exit(EXIT_FAILURE);
    };
    break;
  }
  default:
    break;
  }

  // Sort the result with lowest distance first
  // output the top N to the screen
  sort(resultList.begin(), resultList.end());
  int limit = min(topN, static_cast<int>(resultList.size()));
  for (int i = 0; i < limit; i++)
  {
    ResultStruct res = resultList[i];
    std::cout << res.imgName << " " << res.distance << endl;
  }
  exit(EXIT_SUCCESS);
}
