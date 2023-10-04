// build the histogram
// allocate the histogram, 2D single channel floating point array
// initialized to zero
hist = cv::Mat::zeros(cv::Size (histsize, histsize), CV_32FCA1);

// build the histogram
max = 0; //keep track of the largest bucket
// loop over the src image

float B = src.at<cv::Vec3b>(i, j)[0]

// Compute rg standard chromaticity
float r = R / (R + G + B)

// compute indexes
int rindex = (int)(r * (histsize - 1) + 0.5)

// increment the histogram
hist.at<float>(rindex, gindex) ++;
// update max
float newVal = hist.at<float>(rindex, gindex);

// Histogram completed, need to normalize it
hist /= max: // divides whole Mat by max
// Histogram is all in the range(0, 1)

// Recreate 
dst.create(hist.size(), CV_8UC3);
