/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   : Contain all required methods to compute the features for Project 3
*/
#include "features.h";

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
 * Input : topN, return only the top N largest region found, default is 1
 * Output: List of connected components' region map stats
 */
int getConnectedComponentRegions(cv::Mat &src, vector<cv::Mat> &regionStatList, bool showCC, int topN)
{
    try
    {
        Mat labelImage(src.size(), CV_32S);
        Mat stats;
        Mat centroids;
        // Get connected components
        int nLabels = cv::connectedComponentsWithStats(src, labelImage, stats, centroids, 8);
        vector<RegionStruct> regionList;
        // Go through each found components
        for (int label = 1; label < nLabels; ++label)
        {
            // Check if region valid using the stats
            Mat regionStat = stats.row(label);
            if (isRegionValid(regionStat, src.rows - 1, src.cols - 1))
            {
                // push the region label and statistic to the temp list
                regionList.push_back(RegionStruct(label, regionStat));
            }
        }

        // Sort the result with largest area first
        sort(regionList.begin(), regionList.end());
        // minimum return one result
        int limit = cv::max<int>(1, topN);
        // max is size of region list
        limit = cv::min<int>(limit, regionList.size());

        // Return only the topN region list
        for (int i = 0; i < limit; i++)
        {
            regionStatList.push_back(regionList[i].regionStat);
        }

        // If showCC, labeled the connected component with color and output to the image display
        if (showCC)
        {
            // Initialize all to be background color
            std::vector<Vec3b> colors(nLabels);
            for (int label = 0; label < nLabels; label++)
            {
                colors[label] = Vec3b(0, 0, 0);
            }

            for (int i = 0; i < limit; i++)
            {
                int label = regionList[i].regionId;
                colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
            }

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
 * Compute the scale, translational, and rotational invariant features,
 * first use the rotated bounding rectangle with minimum area obtained from region of interest to calculate for
 * percentage filled, and bounding rectangle width/height
 * Next use the cv::moments() and cv::HuMoments to get the 7 variants of Hu moments
 * Input: Mat of src image , after thresholding and clean up
 * Input: regionStatList for major region identified
 * Output: the nine features computed for each region appended to the featuresList
 * flag: showAR, default is false, if set to true, will show the rotated bounding rectangle and
 * axis of least moment
 */
int computeFeatures(cv::Mat &src, vector<cv::Mat> regionStatList, vector<vector<double>> &featuresList, bool showAR)
{
    try
    {
        Mat displayImg;
        if (showAR)
        {
            cvtColor(src, displayImg, cv::COLOR_GRAY2BGR);
        }

        for (int id = 0; id < regionStatList.size(); id++)
        {
            // Get the region stats
            int leftCorner = regionStatList[id].at<int>(CC_STAT_LEFT);
            int topCorner = regionStatList[id].at<int>(CC_STAT_TOP);
            int rightCorner = leftCorner + regionStatList[id].at<int>(CC_STAT_WIDTH) - 1;
            int btmCorner = topCorner + regionStatList[id].at<int>(CC_STAT_HEIGHT) - 1;

            // Extract the region map
            Mat regionMap = Mat(src, Range(topCorner, btmCorner + 1), Range(leftCorner, rightCorner + 1));

            // Find the contour using the regionMap
            vector<vector<Point>> contours;
            findContours(regionMap, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

            // Use the contours to get the rotated Bounding rectangle
            // there could be more than 1 contours returned, use the largest one
            RotatedRect boundingRect;
            vector<Point> maxContour;
            int maxSize = 0;
            for (int i = 0; i < contours.size(); i++)
            {
                RotatedRect tempRect = cv::minAreaRect(contours[i]);
                if (tempRect.size.width * tempRect.size.height > maxSize)
                {
                    maxSize = tempRect.size.width * tempRect.size.height;
                    boundingRect = tempRect;
                    maxContour = contours[i];
                }
            }

            // Use the contour to get the moments
            Moments regionM = cv::moments(maxContour, true);

            // Building the output vector for the region stats
            vector<double> features;
            // Get the percentFilled and width/height ratio
            float rectWidth = cv::min<float>(boundingRect.size.width, boundingRect.size.height);
            float rectHeight = cv::max<float>(boundingRect.size.width, boundingRect.size.height);
            float boundArea = rectWidth * rectHeight;
            double whratio = rectWidth / rectHeight;
            double percentFilled = regionM.m00 / boundArea;

            features.push_back(percentFilled);
            features.push_back(whratio);

            // Get the Hu Moments
            double huMoments[7];
            cv::HuMoments(regionM, huMoments);
            // apply log scale to Hu Moments and push it to output vector
            for (int i = 0; i < 7; i++)
            {
                huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]));
                features.push_back(huMoments[i]);
            }

            // Add the centroids x and y and push it back to output vector featuresList
            // Get the centroid using moment, add the offset from leftCorner / topCorner
            Point2f centroid;
            centroid.x = regionM.m10 / regionM.m00 + leftCorner;
            centroid.y = regionM.m01 / regionM.m00 + topCorner;
            features.push_back(centroid.x);
            features.push_back(centroid.y);
            featuresList.push_back(features);

            // These sections of codes for showing the AR axis
            if (showAR)
            {
                // Debug code
                /*cout << "rect width: " << rectWidth << endl;
                cout << "rect height: " << rectHeight << endl;
                cout << "rect area: " << boundArea << endl;
                cout << "filled area: " << regionM.m00 << endl;
                cout << "percent filled: " << percentFilled << endl;
                printf("The ratio is %f \n", whratio);*/

                // Get the vertices from the rotated bounding rectangle
                Point2f vertices[4];
                boundingRect.points(vertices);

                // Update it to the original image
                for (int i = 0; i < 4; i++)
                {
                    vertices[i].y += topCorner;
                    vertices[i].x += leftCorner;
                }

                // Draw the rectangle lines on the display image
                for (int i = 0; i < 4; i++)
                {
                    line(displayImg, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
                }

                // Plot the centroid circle on displayImg with color blue
                cv::circle(displayImg, centroid, 5, Scalar(255, 0, 0), FILLED);

                // Get the central axis angle
                double theta = 0.5 * atan(2 * regionM.mu11 / (regionM.mu20 - regionM.mu02));

                // We need to check if theta is within 0 - 90 deg region or 90 - 180 deg
                // by visualization, it should aligned with the longest side of the rotated rectangle
                if (boundingRect.size.width > boundingRect.size.height)
                {
                    theta += PI / 2;
                }

                double thetaD = theta / PI * 180;

                // We need two points to plot the axis of least 2nd moment
                // since the axis will be mostly aligned with the height of the rotated rectangle,
                // we will use this height to extrapolate the points for top right and btm left
                // assuming theta is between 0 - 90 deg
                // cos(theta) give the x offset, sin(theta) give the y offset
                // if theta is 90-180 deg, then cos(theta) will give negative offset, where
                // calculating topright actually give us topleft, and bottomleft actually give us bottom right
                // for top right, it will be centroid.x + x offset, centroid.y - y offset
                Point2f topAxis;
                topAxis.x = centroid.x + rectHeight * cos(theta) / 2;
                topAxis.y = centroid.y - rectHeight * sin(theta) / 2;

                // for btm left, it will be centroid.x + x offset, centroid.y - y offset
                Point2f btmAxis;
                btmAxis.x = centroid.x - rectHeight * cos(theta) / 2;
                btmAxis.y = centroid.y + rectHeight * sin(theta) / 2;

                // Plot it on the display img with color red
                line(displayImg, topAxis, btmAxis, Scalar(0, 0, 255), 2);
            }
        }

        if (showAR)
        {
            imshow("with rotated rounded rectangle", displayImg);
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return 0;
}

/**
 * Main function to process the image to detect the object
 * and compute the feature vectors for the object
 * input : Mat of src image
 * Output: Vector of features for each component
 * input : topN, return only the top N largest region found, default is 1
 * flag  : showInterim, default=false, if it is true, will show images after each preprocessing
 * flag  : showThreshold, default=false, if it is true, will show images after thresholding
 * flag  : showMorpho, default=false, if it is true, will show images after morphological operation
 * flag  : showCC, default=false, if it is true, will show the colored connected components
 * flag  : showAR, default=false, if it is true, will show the axis for least 2nd moment and rotated bounding rectangle of minimum area
 */
int processNCompute(cv::Mat &src, vector<vector<double>> &features, int topN, bool showInterim, bool showThreshold, bool showMorpho, bool showCC, bool showAR)
{
    try
    {
        // Task 1 : Threshold the input frame
        Mat imgThresholded;
        if (thresholdFilter(src, imgThresholded, showInterim) != 0)
        {
            return (-1);
        }
        if (showThreshold)
        {
            imshow("threshold image", imgThresholded);
        }

        // Task 2 : Clean up the binary image
        Mat imgMorphed;
        if (morphologyFilter(imgThresholded, imgMorphed) != 0)
        {
            return (-1);
        }
        if (showMorpho)
        {
            imshow("morpho image", imgMorphed);
        }

        // Task 3 : Segment the image into regions
        // The stats for each major region are recorded in regionStatList
        vector<cv::Mat> regionStatList;
        if (getConnectedComponentRegions(imgMorphed, regionStatList, showCC, topN) != 0)
        {
            return (-1);
        }

        // Task 4 : Compute features for each major region
        if (computeFeatures(imgMorphed, regionStatList, features, showAR) != 0)
        {
            return (-1);
        }
    }
    catch (exception)
    {
        return (-1);
    }
    return 0;
}

/**
 * Helper method to extra the feature texts
 * input : vector of computed features
 * output: vector of size 3 for 3 line of text
 */
int getFeatureText(vector<double> &features, vector<string> &texts)
{
    try
    {
        string temp = "";
        temp += "filled:" + to_string(features[F_PERCENT_FILLED]);
        temp += " whratio: " + to_string(features[F_WIDTH_HEIGHT_Ratio]);
        texts.push_back(string(temp));
        string temp1 = "";
        temp1 += "Hu's 1st:" + to_string(features[F_HU_1ST_MOMENT]);
        temp1 += " Hu's 2nd:" + to_string(features[F_HU_2ND_MOMENT]);
        texts.push_back(string(temp1));
        string temp2 = "";
        temp2 += "Hu's 3rd:" + to_string(features[F_HU_3RD_MOMENT]);
        temp2 += " Hu's 4th:" + to_string(features[F_HU_4TH_MOMENT]);
        texts.push_back(string(temp2));
        string temp3 = "";
        temp3 += "Hu's 5th:" + to_string(features[F_HU_5TH_MOMENT]);
        temp3 += " Hu's 6th:" + to_string(features[F_HU_6TH_MOMENT]);
        texts.push_back(string(temp3));
        string temp4 = "";
        temp4 += " Hu's 7th moment: " + to_string(features[F_HU_7TH_MOMENT]);
        texts.push_back(string(temp4));
    }
    catch (exception)
    {
        return (-1);
    }
    return 0;
}
