/*
  Shiang Jin, Chin

  Other Utilities function used for project 2
  This include
  function to read the all the image files in a directory and return a list of string containing all the valid file name
 */
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <string>
#include <vector>
using namespace std;

/**
 *Given a directory name where the image files reside, this function
  returns check through each of the file, and store the absolute path
  of all image file in the imgLists vector.

  The function returns a non-zero value if something goes wrong.
*/
int readImgFiles(char *dirname, vector<string> &imgLists);

#endif