/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   : Contain all required methods other than filters for Project 3
*/
#include "utils.h";

/**
 * Return a vector of threshold values use to distinguish
 * foreground and background
 * Input: src image of grayscale in one channel
 * Output: single threshold value
 */
int getThreshold(cv::Mat &src, vector<int> &threshold)
{
    try
    {
        Mat centers;
        if (getKMeansCenters(src, centers) != 0)
        {
            return (-1);
        }
        int small = centers.at<float>(0, 0);
        int large = centers.at<float>(1, 0);
        int mid = (int)(small + large) / 2;
        cout << "small:" << small << " large:" << large << " mid:" << mid << endl;
        threshold.push_back(mid);
    }
    catch (exception)
    {
        return (-1);
    }

    return 0;
}

/**
 * Retrieve the centers calculated using kmeans algorithm
 * of two clusters
 */
int getKMeansCenters(cv::Mat &src, cv::Mat &centers)
{
    try
    {
        // Get the row and col size for half of the image
        int newRows = src.rows % 2 == 0 ? src.rows / 2 : src.rows / 2 + 1;
        int newCols = src.cols % 2 == 0 ? src.cols / 2 : src.cols / 2 + 1;

        // Initialize the samples mat to be passed to the kmeans algorithm
        Mat samples(newRows * newCols, 1, CV_32F);
        cout << "samples size, rows: " << samples.rows << " cols:" << samples.cols << endl;

        // Fill the sample
        for (int y = 0; y < src.rows; y += 2)
        {
            int colSum = y / 2 * newCols;
            for (int x = 0; x < src.cols; x += 2)
            {
                samples.at<float>(colSum + x / 2) = src.at<uint8_t>(y, x);
            }
        }
        int clusterCount = 2;
        Mat labels;
        int attempts = 2;
        cv::TermCriteria tc = cv::TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);
        cv::kmeans(samples, clusterCount, labels, tc, attempts, cv::KMEANS_PP_CENTERS, centers);
    }
    catch (exception)
    {
        return (-1);
    }

    return 0;
}

/**
 * Helper function to check if the target region is what we looking for
 * input: Mat regionStats for particular region extracted from cv::connectedComponent
 * return: boolean indicator if the region is target
 */
bool isRegionValid(cv::Mat &regionStats, int btm, int right)
{
    int leftCorner = regionStats.at<int>(CC_STAT_LEFT);
    int topCorner = regionStats.at<int>(CC_STAT_TOP);
    int rightCorner = leftCorner + regionStats.at<int>(CC_STAT_WIDTH) - 1;
    int btmCorner = topCorner + regionStats.at<int>(CC_STAT_HEIGHT) - 1;
    // return false if it border the boundary
    if (left <= 0 || topCorner <= 0 || rightCorner >= right || btmCorner >= btm)
    {
        return false;
    }

    // return false if component area is too small (<200)
    if (regionStats.at<int>(CC_STAT_AREA) < 200)
    {
        return false;
    }
    return true;
}

/**
 * Get the connected components using standard openCV connectedComponentsWithStats function
 * Labeled the connected region and show it in a window
 * Input: Mat of src image
 * Output: List of connected components' region map
 */
int getConnectedComponentRegions(cv::Mat &src, vector<cv::Mat> &regionList, bool showCC)
{
    try
    {
        Mat labelImage(src.size(), CV_32S);
        Mat stats;
        Mat centroids;
        int nLabels = cv::connectedComponentsWithStats(src, labelImage, stats, centroids, 8);
        std::vector<Vec3b> colors(nLabels);
        colors[0] = Vec3b(0, 0, 0); // background
        cout << "total labels: " << nLabels << endl;

        // Go through each found components
        for (int label = 1; label < nLabels; ++label)
        {
            // Check if region valid using the stats
            Mat regionStat = stats.row(label);
            if (isRegionValid(regionStat, src.rows - 1, src.cols - 1))
            {
                // assign random color to that region
                colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
                // Push the region map to regionList
                // extract the region map
                int top = stats.at<int>(label, CC_STAT_TOP);
                int height = stats.at<int>(label, CC_STAT_HEIGHT);
                int left = stats.at<int>(label, CC_STAT_LEFT);
                int width = stats.at<int>(label, CC_STAT_WIDTH);
                Mat regionMap = Mat(src, Range(top, top + height), Range(left, left + width));
                regionList.push_back(regionMap);
            }
            else
            {
                // give it background image
                colors[label] = Vec3b(0, 0, 0);
            }
        }

        /* //Debug code to print the stats
        for (int label = 0; label < nLabels; ++label)
        {
            cout << " stats for label: " << label << endl;
            cout << " stats size" << stats.size() << endl;
            cout << " CC_STAT_LEFT " << stats.at<int>(label, CC_STAT_LEFT);
            cout << " CC_STAT_TOP " << stats.at<int>(label, CC_STAT_TOP);
            cout << " CC_STAT_WIDTH " << stats.at<int>(label, CC_STAT_WIDTH);
            cout << " CC_STAT_HEIGHT " << stats.at<int>(label, CC_STAT_HEIGHT);
            cout << " CC_STAT_AREA " << stats.at<int>(label, CC_STAT_AREA) << endl;
        } */

        // If showCC, labeled the connected component and output to the image display
        if (showCC)
        {
            Mat imgDisplay(src.size(), CV_8UC3);
            for (int r = 0; r < imgDisplay.rows; ++r)
            {
                for (int c = 0; c < imgDisplay.cols; ++c)
                {
                    int label = labelImage.at<int>(r, c);
                    Vec3b &pixel = imgDisplay.at<Vec3b>(r, c);
                    pixel = colors[label];
                }
            }
            imshow("Connected Components", imgDisplay);
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return 0;
}

/**
 * Using the rotated bounding rectangle with minimum area obtained from region of interest
 * calculate the scale, translational, and rotational invariant for
 * percentage filled, and bounding rectangle width/height
 * Input: Mat of src image
 * Output: the two features computed appended to the features list
 * flag: showAxis default is false, if set to true, will draw the axis for least 2nd moment for the object
 */
int getOrientedBoundingBoxStat(cv::Mat &src, vector<float> &features, bool showAxis)
{

    Mat displayImg = src.clone();
    vector<vector<Point>> contours;
    findContours(src, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    RotatedRect boundingRect = cv::minAreaRect(contours[0]);
    cout << "size of rect: " << boundingRect.size << endl;
    Point2f vertices[4];
    boundingRect.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        line(displayImg, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
    }
    imshow("with rotated rounded rectangle", displayImg);
    return 0;
}