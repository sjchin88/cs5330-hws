/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/06/2023
  Description   : Other Utilities function used for project 2. This include
                  function to read the all the image files in a directory and
                  return a list of string containing all the valid file name
*/

#include "utils.h";

/**
 *Given a directory name where the image files reside, this function
  returns check through each of the file, and store the absolute path
  of all image file in the imgLists vector.

  The function returns a non-zero value if something goes wrong.
*/
int readImgFiles(char *dirname, vector<string> &imgLists)
{
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    char buffer[256];

    printf("Processing directory %s\n", dirname);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL)
    {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif"))
        {

            // printf("processing image file: %s\n", dp->d_name);

            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            // printf("full path name: %s\n", buffer);
            string temp(buffer);
            // cout << temp << endl;
            imgLists.push_back(temp);
        }
    }

    printf("Finish Processing\n");

    return (0);
}