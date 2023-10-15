#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <ctime>
#include "filters.h"
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    VideoCapture *capdev;

    // Set the cameraIdx from the commandline argument
    int cameraIdx;
    try
    {
        cameraIdx = std::stoi(argv[1]);
    }
    catch (std::exception)
    {
        // Set it to 0
        cameraIdx = 0;
    }
    cout << "camera idx is " << cameraIdx << endl;

    // open the video device
    capdev = new cv::VideoCapture(cameraIdx);
    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        return (-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);
    int fps = (int)capdev->get(cv::CAP_PROP_FPS);
    printf("Captured fps: %d \n", fps);

    // Set default fps for saving
    int savingFPS = 30;
    if (fps > 0)
    {
        savingFPS = fps;
    }

    // Set default save path
    String defaultSavePath;
    try
    {
        defaultSavePath = argv[2];
    }
    catch (std::exception)
    {
        std::cout << "Could not read the default save path " << std::endl;
        return 1;
    }
    cout << "save path is " << defaultSavePath << endl;

    // Set the caption text
    String captionText;
    try
    {
        captionText = argv[3];
    }
    catch (std::exception)
    {
        // set it to default text
        std::cout << "Unable to read caption text argument, setting it to default " << std::endl;
        captionText = "Hahaha";
    }
    cout << "caption text is " << captionText << endl;

    // Initialize variables required
    namedWindow("Video", 1); // identifies a window
    Mat capturedFrame;
    Mat tempFrame;
    Mat displayFrame;
    char selection = 'R';
    bool captioning = false;
    bool videoSaving = false;
    cv::VideoWriter videoWriter;

    // Run the while loop
    while (true)
    {
        *capdev >> capturedFrame; // get a new frame from the camera, treat as a stream

        if (capturedFrame.empty())
        {
            printf("frame is empty\n");
            break;
        }

        // Default is to set the displayFrame as capturedFrame
        displayFrame = capturedFrame;

        // Try to apply the selected effect to the capturedFrame
        // and set the displayFrame to modified frame only if successful
        switch (selection)
        {
        case 'G':
            // Task3: Use the cvtColor() function to grayscale the image
            cvtColor(capturedFrame, displayFrame, COLOR_BGR2GRAY);
            break;

        case 'H':
            if (greyscale(capturedFrame, tempFrame) == 0)
            {
                displayFrame = tempFrame;
            }
            break;

        case 'B':
            if (blur5x5(capturedFrame, tempFrame) == 0)
            {
                displayFrame = tempFrame;
            }
            break;

        case 'X':
            if (sobelX3x3(capturedFrame, tempFrame) == 0)
            {
                // Output of sobelX3x3 filter are stored in 16S, convert it for display
                Mat tempFrameOut;
                convertScaleAbs(tempFrame, tempFrameOut);
                displayFrame = tempFrameOut;
            }
            break;

        case 'Y':
            if (sobelY3x3(capturedFrame, tempFrame) == 0)
            {
                // Output of sobelY3x3 filter are stored in 16S, convert it for display
                Mat tempFrameOut;
                convertScaleAbs(tempFrame, tempFrameOut);
                displayFrame = tempFrameOut;
            }
            break;

        case 'M':
        {
            Mat tempFrameX;
            Mat tempFrameY;
            Mat tempFrameM;
            Mat tempFrameOut;
            if (sobelX3x3(capturedFrame, tempFrameX) == 0 && sobelY3x3(capturedFrame, tempFrameY) == 0)
            {
                if (magnitude(tempFrameX, tempFrameY, tempFrameM) == 0)
                {
                    // Output of magnitude filter are stored in 16S, convert it for display
                    convertScaleAbs(tempFrameM, tempFrameOut);
                    displayFrame = tempFrameOut;
                }
            }
            break;
        }

        case 'I':
        {
            int defaultLevel = 15;
            if (blurQuantize(capturedFrame, tempFrame, defaultLevel) == 0)
            {
                displayFrame = tempFrame;
            }
        }

        case 'C':
        {
            int defaultLevel = 15;
            int defaultMagThreshold = 15;
            auto start = std::chrono::steady_clock::now();
            if (cartoon(capturedFrame, tempFrame, defaultLevel, defaultMagThreshold) == 0)
            {
                displayFrame = tempFrame;
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
            std::cout << "time taken to process a frame: " << elapsed.count() << "ms" << endl;
            break;
        }

        case 'N':
            if (negative(capturedFrame, tempFrame) == 0)
            {
                displayFrame = tempFrame;
            }
            break;

        default:
            displayFrame = capturedFrame;
            break;
        }

        // check if captioning is on, and add the caption text
        if (captioning)
        {
            cv::Point2i btmLeft(50, 100);
            cv::putText(displayFrame, captionText, btmLeft, cv::FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0));
        }

        // check if video saving is on
        if (videoSaving)
        {
            videoWriter.write(displayFrame);
        }

        cv::imshow("Video", displayFrame);

        // see if there is a waiting keystroke every 33ms
        char key = cv::waitKey(33);

        switch (key)
        {

        // Task 3: keystroke g turn the current frame into greyscale
        case 'g':
            selection = 'G';
            break;

        // Task 4: keystroke h turn the current frame into custom greyscale
        case 'h':
            selection = 'H';
            break;

        // Task 5: keystroke h turn the current frame into blur5x5
        case 'b':
            selection = 'B';
            break;

        // Task 6: keystroke x, y turn the current frame into Sobel Filter
        case 'x':
            selection = 'X';
            break;

        case 'y':
            selection = 'Y';
            break;

        // Task 7: keystroke m turn the current frame into gradient magnitude image
        case 'm':
            selection = 'M';
            break;

        // Task 8: keystroke i turn the current frame into blurs and quantizes image
        case 'i':
            selection = 'I';
            break;

        // Task 9: keystroke c turn the current frame into cartoon image
        case 'c':
            selection = 'C';
            break;

        // Task 10: keystroke n turn the current frame into negative of itself
        case 'n':
            selection = 'N';
            break;

        // Extension 1: keystroke t to turn on/off caption text
        case 't':
            captioning = !captioning;
            break;

        // Extension 2: keystroke v turn on saving video to the file
        case 'v':
        {
            // if videoSaving is on, turn it off
            if (videoSaving)
            {
                videoWriter.release();
                videoSaving = false;
            }
            else
            {
                // Set the video writer
                std::time_t timeStamp = std::time(nullptr);
                String videoPath = defaultSavePath + "video_" + to_string(timeStamp) + ".avi";
                /*  cv::VideoWriter::VideoWriter (const String &filename, int fourcc, double fps,Size frameSize,bool isColor = true
                        )	 */
                videoWriter = cv::VideoWriter(videoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), savingFPS, refS);
                videoSaving = true;
            }

            break;
        }

        // keystroke r to change the selection back to normal
        case 'r':
            selection = 'R';
            break;

        default:
            break;
        }

        // Task 2: keystroke s save the current display frame
        if (key == 's')
        {
            std::time_t timeStamp = std::time(nullptr);
            String finalPath = defaultSavePath + "screenshot_" + to_string(timeStamp) + ".png";
            imwrite(finalPath, displayFrame);
        }

        // keystroke q quit the program
        if (key == 'q')
        {
            break;
        }
    }

    delete capdev;
    return (0);
}