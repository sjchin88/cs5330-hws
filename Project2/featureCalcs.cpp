/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/06/2023
  Description   : All calculators to convert given image into a set of feature vectors for project 2
*/
#include "featureCalcs.h";
/**
 * Calculate baseline feature vectors from the src and append them into the featureVectors
 * by extracting the 9 x 9 square in the middle of the image
 */
int calcBaseline(cv::Mat &src, vector<int> &featureVectors)
{
    try
    {
        // Get the number of rows and cols from the source
        const int rows = src.rows;
        const int cols = src.cols;

        // calculate the start & end indexes for rows and cols
        // if rows size is 15, middle is 7, it will start at idx 3
        const int rowStart = rows / 2 - 4;
        const int rowEnd = rowStart + 8;
        const int colStart = cols / 2 - 4;
        const int colEnd = colStart + 8;
        for (int y = rowStart; y <= rowEnd; y++)
        {
            for (int x = colStart; x <= colEnd; x++)
            {
                Vec3b bgrPixel = src.at<cv::Vec3b>(y, x);
                // flatten three color channels and push to the vector
                featureVectors.push_back(bgrPixel[0]);
                featureVectors.push_back(bgrPixel[1]);
                featureVectors.push_back(bgrPixel[2]);
            }
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return (0);
}

/**
 * Calculate the feature vectors from the src and append them into the featureVectors
 * based on rg chromaticity histogram
 * the histSize can be set by the user
 */
int calcRGHist(cv::Mat &src, vector<float> &featureVectors, const int histSize)
{
    try
    {
        // Initialize the histogram with Mat of type 2D single channel floating array
        Mat histogram = Mat::zeros(cv::Size(histSize, histSize), CV_32FC1);

        // Loop through original src matrix, and update the histogram count
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                float B = src.at<cv::Vec3b>(i, j)[0];
                float G = src.at<cv::Vec3b>(i, j)[1];
                float R = src.at<cv::Vec3b>(i, j)[2];
                float total = (R + G + B);
                float r;
                float g;

                // Handle black pixel, where RGB total is < 15
                // set it to grey value
                if (total < 15)
                {
                    r = 1.0 / 3;
                    g = 1.0 / 3;
                }
                else
                {
                    r = R / total;
                    g = G / total;
                }

                // Get r and g index and increment the histogram
                // Formula below effectively convert r to scale of 0 : histSize - 1
                // if histSize = 8, then we have scale of 0 to 7
                // if value is 0.75 then 0.75 * 8 - 0.000001 = 5
                // -0.000001 to account for border case (r/g == 1)
                int ridx = r == 0 ? 0 : (int)(r * histSize - 0.000001);
                int gidx = g == 0 ? 0 : (int)(g * histSize - 0.000001);
                histogram.at<float>(ridx, gidx)++;
            }
        }

        // Normalize the histogram
        histogram /= (src.rows * src.cols);

        // Write histogram to vector
        for (int y = 0; y < histogram.rows; y++)
        {
            for (int x = 0; x < histogram.cols; x++)
            {
                featureVectors.push_back(histogram.at<float>(y, x));
            }
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return (0);
}

/**
 * Calculate the feature vectors based on R, G, B histogram of 8 bins each frpm the src
 * flatten the normalized histogram values into the 1-D feature vectors
 * and append them into the featureVectors
 */
int calcRGBHist(cv::Mat &src, vector<float> &featureVectors)
{
    try
    {
        int size3D[3] = {8, 8, 8};
        // Initialize the histogram with Mat of type 3D single channel floating array
        Mat histogram = Mat::zeros(3, size3D, CV_32FC1);
        // Loop through original src matrix, and update the histogram count
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                int B = src.at<cv::Vec3b>(i, j)[0];
                int G = src.at<cv::Vec3b>(i, j)[1];
                int R = src.at<cv::Vec3b>(i, j)[2];

                // Get r, g, b index in 8 bins by
                // dividing R/G/B by 32
                int ridx = R / 32;
                int gidx = G / 32;
                int bidx = B / 32;
                histogram.at<float>(ridx, gidx, bidx)++;
            }
        }

        // Normalize the histogram
        histogram /= (src.rows * src.cols);

        // Write histogram to vector
        for (int z = 0; z < 8; z++)
        {
            for (int y = 0; y < 8; y++)
            {
                for (int x = 0; x < 8; x++)
                {
                    featureVectors.push_back(histogram.at<float>(z, y, x));
                }
            }
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return (0);
}

/**
 * Calculate the feature vectors based on two rg chromaticity histogram
 * for the left and right part of the src image
 * Output the normalized histogram values into the feature vectors
 * the histSize can be set by the user
 */
int calcMultiHistLR(cv::Mat &src, vector<float> &featureVectors, const int histSize)
{
    try
    {
        // Get the number of rows and cols from the source
        const int rows = src.rows;
        const int cols = src.cols;

        Mat left_image = src(Range(0, rows), Range(0, cols / 2));
        Mat right_image = src(Range(0, rows), Range(cols / 2, cols));
        // Compute rg chromaticity histogram of left image
        if (calcRGHist(left_image, featureVectors, histSize) != 0)
        {
            return (-1);
        }
        // Compute rg chromaticity histogram of right image
        if (calcRGHist(right_image, featureVectors, histSize) != 0)
        {
            return (-1);
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return (0);
}

/**
 * Helper method to calculate the whole image texture histogram
 * using gradient magnitude filters
 */
int calcGradTexture(cv::Mat &src, vector<float> &featureVectors)
{
    try
    {
        Mat tempImgX;
        Mat tempImgY;
        Mat tempImgM;
        Mat tempImgTexture;
        // First use the sobelX filtered image and sobelY filtered Image
        if (sobelX3x3(src, tempImgX) == 0 && sobelY3x3(src, tempImgY) == 0)
        {
            // To compute the magnitude
            if (magnitude(tempImgX, tempImgY, tempImgM) == 0)
            {
                // Output of magnitude filter are stored in 16S, convert it
                convertScaleAbs(tempImgM, tempImgTexture);
            }
            else
            {
                return (-1);
            }
        }
        else
        {
            return (-1);
        }
        // add the gradient magnitude image vector
        if (calcRGBHist(tempImgTexture, featureVectors) != 0)
        {
            return (-1);
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return (0);
}

/**
 * Helper method to get the focus image from src
 * based on zoom factor
 * and store it in dst
 */
int getFocusImg(cv::Mat &src, cv::Mat &dst, float zoomFactor)
{
    try
    {
        if (zoomFactor < 1.0)
        {
            // Get the number of rows and cols from the source
            const int rows = src.rows;
            const int cols = src.cols;

            // calculate the image boundary based on zoom factor
            const int top = rows / 2 - (rows * zoomFactor) / 2;
            const int bottom = top + (rows * zoomFactor);
            const int left = cols / 2 - (cols * zoomFactor) / 2;
            const int right = left + (cols * zoomFactor);

            // Get the zoom image
            dst = src(Range(top, bottom), Range(left, right));
        }
        else if (zoomFactor == 1.0)
        {
            dst = src;
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return (0);
}

/**
 * Calculate the feature vectors based on a zoom portion of the src image
 * combining color histogram and texture histogram based on gradient magnitude image
 * acceptable zoomFactor is between 0.1 to 1.0 (1.0 is default option and cover whole image)
 * the zoomed image will center around the center of the original image
 * append them into the featureVectors
 */
int calcRGBNTexture(cv::Mat &src, vector<float> &featureVectors, float zoomFactor)
{
    if (zoomFactor < 0.1 || zoomFactor > 1.0)
    {
        cout << "Invalid zoom factor, please choose between 0.1 to 1.0" << endl;
        return (-1);
    }
    try
    {
        Mat focusImg;
        if (getFocusImg(src, focusImg, zoomFactor) != 0)
        {
            return (-1);
        }
        // Get the feature vector
        // First calculate the RGB histogram
        if (calcRGBHist(focusImg, featureVectors) != 0)
        {
            return (-1);
        }
        // Next calculate the texture histogram based on gradient magnitude
        if (calcGradTexture(focusImg, featureVectors) != 0)
        {
            return (-1);
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return (0);
}

/**
 * Helper method to calculate the whole image texture histogram
 * using a set of Gabor filters
 */
int calcGaborTexture(cv::Mat &src, vector<float> &featureVectors, bool smallImg)
{
    try
    {
        Mat tempImgTexture;
        // If it is small image use a smaller kernel for faster computation
        // if it is large image use the larger kernel for faster computation
        cv::Size ksize = smallImg ? cv::Size(11, 11) : cv::Size(31, 31);
        // This list of sigmas, thetas, lambdas, gammas are preset values
        vector<float> sigmas;
        sigmas.push_back(4.0);
        vector<float> thetas;
        for (float i = 0.0; i < CV_PI; i += CV_PI / 16)
        {
            thetas.push_back(i);
        }
        vector<float> lambdas;
        lambdas.push_back(10.0);
        vector<float> gammas;
        gammas.push_back(0.5);

        // Perform filtering
        if (gaborFiltering(src, tempImgTexture, ksize, sigmas, thetas, lambdas, gammas) != 0)
        {
            return (-1);
        }
        // Convert result to RGB histogram
        if (calcRGBHist(tempImgTexture, featureVectors) != 0)
        {
            return (-1);
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return (0);
}

/**
 * Calculate the feature vector based on a zoom portion of the src image
 * combining color histogram and texture histogram based on gabor filters
 * acceptable zoomFactor is between 0.1 to 1.0 (1.0 is default option and cover whole image)
 * the zoomed image will center around the center of the original image
 * append them into the featureVectors
 */
int calcRGBNGabor(cv::Mat &src, vector<float> &featureVectors, float zoomFactor)
{
    if (zoomFactor < 0.1 || zoomFactor > 1.0)
    {
        cout << "Invalid zoom factor, please choose between 0.1 to 1.0" << endl;
        return (-1);
    }
    try
    {
        Mat focusImg;
        if (getFocusImg(src, focusImg, zoomFactor) != 0)
        {
            return (-1);
        }
        // if zoomFactor < 0.6, consider image to be small
        bool smallImg = zoomFactor < 0.6 ? true : false;
        // First calculate the RGB histogram
        if (calcRGBHist(focusImg, featureVectors) != 0)
        {
            return (-1);
        }
        // Next calculate the texture histogram based on Gabor filters
        if (calcGaborTexture(focusImg, featureVectors, smallImg) != 0)
        {
            return (-1);
        }
    }
    catch (exception)
    {
        return (-1);
    }

    return (0);
}