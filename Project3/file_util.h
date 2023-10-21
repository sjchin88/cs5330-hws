/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   : Based on files given in Project2, contains all methods required to
  read and append to csv file, and read the image list from directory
*/

#ifndef FILE_UTIL_H
#define FILE_UTIL_H
#include <cstdio>
#include <cstring>
#include <string>
#include <dirent.h>
#include <vector>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

/*
  Given a filename, and image filename, and the image features, by
  default the function will append a line of data to the CSV format
  file.  If reset_file is true, then it will open the file in 'write'
  mode and clear the existing contents.

  The image filename is written to the first position in the row of
  data. The values in image_data are all written to the file as
  floats.

  The function returns a non-zero value in case of an error.
 */
int append_image_data_csv(char *filename, char *image_filename, std::vector<float> &image_data, int reset_file = 0);

/*
  Given a filename, and image filename, and the image features, by
  default the function will append a line of data to the CSV format
  file.  If reset_file is true, then it will open the file in 'write'
  mode and clear the existing contents.

  The image filename is written to the first position in the row of
  data. The values in image_data are all written to the file as
  int.

  The function returns a non-zero value in case of an error.
 */
int append_image_data_csv(char *filename, char *image_filename, std::vector<int> &image_data, int reset_file = 0);

/*
  Given a file with the format of a string as the first column and
  floating point numbers as the remaining columns, this function
  returns the filenames as a std::vector of character arrays, and the
  remaining data as a 2D std::vector<float>.

  filenames will contain all of the image file names.
  data will contain the features calculated from each image.

  If echo_file is true, it prints out the contents of the file as read
  into memory.

  The function returns a non-zero value if something goes wrong.
 */
int read_image_data_csv(char *filename, std::vector<char *> &filenames, std::vector<std::vector<float>> &data, int echo_file = 0);

/*
  Given a file with the format of a string as the first column and
  floating point numbers as the remaining columns, this function
  returns the filenames as a std::vector of character arrays, and the
  remaining data as a 2D std::vector<int>.

  filenames will contain all of the image file names.
  data will contain the features calculated from each image.

  If echo_file is true, it prints out the contents of the file as read
  into memory.

  The function returns a non-zero value if something goes wrong.
 */
int read_image_data_csv(char *filename, std::vector<char *> &filenames, std::vector<std::vector<int>> &data, int echo_file = 0);

/**
 *Given a directory name where the image files reside, this function
  returns check through each of the file, and store the absolute path
  of all image file in the imgLists vector.

  The function returns a non-zero value if something goes wrong.
  method taken from project2
*/
int readImgFiles(char *dirname, vector<string> &imgLists, vector<string> &objNames);
#endif
