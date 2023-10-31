/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   :
*/

#ifndef OBJFILE_LOADER_H
#define OBJFILE_LOADER_H

#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <dirent.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

// Define a class to hold the polygon face
struct FaceStruct
{
    std::vector<int> vertexesId;
    std::vector<int> textCoordsId;
    std::vector<int> vertexNormalsId;
    FaceStruct(std::vector<int> &vertexes, std::vector<int> &textCoords, std::vector<int> &vertexNorms) : vertexesId(vertexes), textCoordsId(textCoords), vertexNormalsId(vertexNorms) {}
};

struct OBJStruct
{
    std::vector<cv::Vec3f> vertexes;
    std::vector<cv::Vec2f> textCoords;
    std::vector<cv::Vec3f> vertexNorms;
    std::vector<FaceStruct> faces;

    OBJStruct()
    {
    }

    // https://www.opengl-tutorial.org/beginners-tutorials/tutorial-7-model-loading/
    int loadFile(std::string objFilePath)
    {
        // Open and read the obj file
        try
        {
            std::cout << objFilePath << std::endl;
            FILE *objFile = fopen(objFilePath.c_str(), "r");
            if (objFile == NULL)
            {
                printf("Cannot open the targetfile! \n");
                return (-1);
            }

            while (true)
            {
                char lineHeader[128];
                int res = fscanf(objFile, "%s", &lineHeader);
                // std::cout << lineHeader << std::endl;
                if (res == EOF)
                {
                    break;
                }

                if (strcmp(lineHeader, "v") == 0)
                {
                    cv::Vec3f vertex;
                    fscanf(objFile, "%f %f %f\n", &vertex[0], &vertex[1], &vertex[2]);
                    vertexes.push_back(vertex);
                }
                else if (strcmp(lineHeader, "vt") == 0)
                {
                    cv::Vec2f textureCoord;
                    fscanf(objFile, "%f %f \n", &textureCoord[0], &textureCoord[1]);
                    textCoords.push_back(textureCoord);
                }
                else if (strcmp(lineHeader, "vn") == 0)
                {
                    cv::Vec3f normalVector;
                    fscanf(objFile, "%f %f %f\n", &normalVector[0], &normalVector[1], &normalVector[2]);
                    vertexNorms.push_back(normalVector);
                }
                else if (strcmp(lineHeader, "f") == 0)
                {
                    // std::cout << "reading face" << std::endl;
                    unsigned int vertexIds[3], textCoordIds[3], vertexNormIds[3];
                    int matches = fscanf(objFile, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIds[0], &textCoordIds[0], &vertexNormIds[0],
                                         &vertexIds[1], &textCoordIds[1], &vertexNormIds[1], &vertexIds[2], &textCoordIds[2], &vertexNormIds[2]);
                    // std::cout << matches << std::endl;
                    if (matches != 9)
                    {
                        printf("Unable to parse one face");
                        continue;
                    }
                    std::vector<int> vertexIndexes;
                    std::vector<int> textCoordIndexes;
                    std::vector<int> vertexNormalIndexes;
                    vertexIndexes.push_back(vertexIds[0]);
                    vertexIndexes.push_back(vertexIds[1]);
                    vertexIndexes.push_back(vertexIds[2]);
                    textCoordIndexes.push_back(textCoordIds[0]);
                    textCoordIndexes.push_back(textCoordIds[1]);
                    textCoordIndexes.push_back(textCoordIds[2]);
                    vertexNormalIndexes.push_back(vertexNormIds[0]);
                    vertexNormalIndexes.push_back(vertexNormIds[1]);
                    vertexNormalIndexes.push_back(vertexNormIds[2]);
                    // std::cout << "read ok" << std::endl;
                    FaceStruct face(vertexIndexes, textCoordIndexes, vertexNormalIndexes);
                    faces.push_back(face);
                }
            }
            printf("Obj read completed");
        }
        catch (std::exception e)
        {
            std::cout << e.what() << std::endl;
        }

        return 0;
    }

    int projectObject(cv::Mat &dst, cv::Mat &cameraMatrix, cv::Mat &distCoeffs, cv::Mat &rvec, cv::Mat &tvec)
    {
        for (auto face : faces)
        {
            std::vector<cv::Vec3f> tempPoints;
            for (auto vertexId : face.vertexesId)
            {
                cv::Vec3f tempPoint = vertexes[vertexId - 1];
                // Scale it
                tempPoint = cv::Vec3f(-tempPoint[2], tempPoint[0], -tempPoint[1]);
                tempPoint *= 3;
                tempPoints.push_back(tempPoint);
            }
            // Get the projection
            std::vector<cv::Point2f> arPoints;
            // std::vector<cv::Point2i> arPoints;
            cv::projectPoints(tempPoints, rvec, tvec, cameraMatrix, distCoeffs, arPoints);
            std::vector<cv::Point2i> polyPoints;
            for (auto point : arPoints)
            {
                cv::Point2i temp;
                temp.x = (int)point.x;
                temp.y = (int)point.y;
                polyPoints.push_back(temp);
            }
            cv::fillConvexPoly(dst, polyPoints, cv::Scalar(0, 128, 255));
            cv::polylines(dst, polyPoints, true, cv::Scalar(0, 0, 0));
        }
        return 0;
    }
};

#endif
