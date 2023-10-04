#include "utils.h";

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

    printf("Terminating\n");

    return (0);
}