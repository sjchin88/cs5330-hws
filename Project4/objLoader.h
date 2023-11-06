/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 11/03/2023
  Description   : Contains Struct class and method required to load 3-D object from .obj and/or .mtl file
                and display it onto the scene containing chessboard pattern
*/

#ifndef OBJFILE_LOADER_H
#define OBJFILE_LOADER_H

#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <dirent.h>
#include <vector>
#include <map>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

// Define a class to hold the polygon face
// Which contain the vertexesId, textCoordsId, vertexNormalsId
// and the color in scalar (B, G, R) format
struct FaceStruct
{
    std::vector<int> vertexesId;
    std::vector<int> textCoordsId;
    std::vector<int> vertexNormalsId;
    cv::Scalar color;
    FaceStruct(std::vector<int> &vertexes, std::vector<int> &textCoords, std::vector<int> &vertexNorms, cv::Scalar &colorBGR) : vertexesId(vertexes), textCoordsId(textCoords), vertexNormalsId(vertexNorms), color(colorBGR) {}
};

/*
 * Define a class that hold the 3-D objects information
 * with members included the vertexes, texture coordinates, vertex normals
 * color map for each material used, and faces that store all the polygon face
 * boolean member colored = false will use the default color provided
 * Other wise it will try to load the material face color from .mtl file
 */
struct OBJStruct
{
    std::vector<cv::Vec3f> vertexes;
    std::vector<cv::Vec2f> textCoords;
    std::vector<cv::Vec3f> vertexNorms;
    std::map<std::string, cv::Scalar> colorMap;
    std::vector<FaceStruct> faces;
    float objWidth = 0;
    float objCenterX = 0;
    float objCenterY = 0;
    bool colored = false;

    /**
     * Default Constructor
     */
    OBJStruct(cv::Scalar &defaultColor)
    {
        colorMap["default"] = defaultColor;
    }

    /**
     * Load the file
     */
    int loadFile(std::string &objFilePath, bool coloredObj = false, std::string mtlFilePath = "")
    {
        colored = coloredObj;
        std::cout << objFilePath << std::endl;
        if (coloredObj)
        {
            loadMtlFile(mtlFilePath);
        }
        loadObjFile(objFilePath, coloredObj);
        return 0;
    }

    /**
     * Load the mtl file
     */
    int loadMtlFile(std::string &mtlFilePath)
    {
        try
        {
            FILE *mtlFile = fopen(mtlFilePath.c_str(), "r");
            if (mtlFile == NULL)
            {
                printf("Cannot open the targetfile! \n");
                return (-1);
            }

            std::string mtlName;
            while (true)
            {
                char lineHeader[128];
                int res = fscanf(mtlFile, "%s", &lineHeader);
                // std::cout << lineHeader << std::endl;
                if (res == EOF)
                {
                    break;
                }

                // detect new material
                if (strcmp(lineHeader, "newmtl") == 0)
                {
                    fscanf(mtlFile, "%s\n", &lineHeader);
                    mtlName = std::string(lineHeader);
                    std::cout << "material name" << mtlName << std::endl;
                }

                // we only interest in kd at this moment
                else if (strcmp(lineHeader, "Kd") == 0 || strcmp(lineHeader, "kd") == 0)
                {
                    cv::Vec3f color;
                    fscanf(mtlFile, "%f %f %f\n", &color[0], &color[1], &color[2]);
                    cv::Scalar colorBGR(color[2] * 255, color[1] * 255, color[0] * 255);
                    std::cout << "color: " << colorBGR << std::endl;
                    colorMap[mtlName] = colorBGR;
                }
            }
            // Print the loaded material color
            for (auto const &x : colorMap)
            {
                std::cout << x.first << ":" << x.second << std::endl;
            }
        }
        catch (std::exception e)
        {
            std::cerr << e.what() << std::endl;
        }

        return 0;
    }

    /**
     * Load the .obj file
     */
    int loadObjFile(std::string &objFilePath, bool coloredObj = false)
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

            // place holder for current material name
            std::string mtlName;
            while (true)
            {
                char lineHeader[128];
                int res = fscanf(objFile, "%s", &lineHeader);
                if (res == EOF)
                {
                    break;
                }

                // detect new material
                if (strcmp(lineHeader, "usemtl") == 0)
                {
                    fscanf(objFile, "%s\n", &lineHeader);
                    mtlName = std::string(lineHeader);
                    std::cout << "material name" << mtlName << std::endl;
                }
                else if (strcmp(lineHeader, "v") == 0)
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
                    // reading the faces
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

                    // Set the color for the face
                    cv::Scalar colorBGR = coloredObj ? cv::Scalar(colorMap[mtlName]) : cv::Scalar(colorMap["default"]);
                    FaceStruct face(vertexIndexes, textCoordIndexes, vertexNormalIndexes, colorBGR);
                    faces.push_back(face);
                }
            }

            // Try to get the center x and y for the 3-D object
            float minX = FLT_MAX;
            float maxX = FLT_MIN;
            float minY = FLT_MAX;
            float maxY = FLT_MIN;
            for (auto vertex : vertexes)
            {
                minX = std::min<float>(minX, vertex[0]);
                maxX = std::max<float>(maxX, vertex[0]);
                minY = std::min<float>(minY, vertex[1]);
                maxY = std::max<float>(maxY, vertex[1]);
            }
            printf("Obj read completed");
            objWidth = maxX - minX;
            objCenterX = minX + (maxX - minX) / 2;
            objCenterY = minY + (maxY - minY) / 2;
            printf(" center is %f, %f", objCenterX, objCenterY);
        }
        catch (std::exception e)
        {
            std::cerr << e.what() << std::endl;
        }

        return 0;
    }

    /**
     * Project the 3-D object, based on
     * inputOutput : Mat dst - target frame
     * input: Mat cameraMatrix -> camera matrix parameter
     * input: Mat distCoeffs -> distortion coefficients parameter
     * input: Mat rvec -> rotation parameters
     * input: Mat tvec -> translation parameters
     * input: scale -> scale factor to be used
     * input: centerX -> center of chessboard in world coordinate
     * input: centerY -> center of chessboard in world coordinate
     */
    int projectObject(cv::Mat &dst, cv::Mat &cameraMatrix, cv::Mat &distCoeffs, cv::Mat &rvec, cv::Mat &tvec, float scaleAR, float centerX, float centerY, float yaw, float pitch, float roll)
    {
        // Set the boolean and skip the transformation if all angles are 0
        bool rotated = (yaw == 0 && pitch == 0 && roll == 0) ? false : true;
        // Intrinsic rotation with Tait-Bryan angles
        float alpha = (yaw)*CV_PI / 180.;
        float beta = (pitch)*CV_PI / 180.;
        float gamma = (roll)*CV_PI / 180.;
        // Rotation matrices around the X, Y, and Z axis
        cv::Mat RX = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                      0, cos(gamma), -sin(gamma),
                      0, sin(gamma), cos(gamma));
        cv::Mat RY = (cv::Mat_<double>(3, 3) << cos(beta), 0, -sin(beta),
                      0, 1, 0,
                      sin(beta), 0, cos(beta));
        cv::Mat RZ = (cv::Mat_<double>(3, 3) << cos(alpha), -sin(alpha), 0,
                      sin(alpha), cos(alpha), 0,
                      0, 0, 1);
        // Composed rotation matrix with (RX, RY, RZ)
        cv::Mat R = RZ * RY * RX;

        // Get the scale factor, using the width of the pattern (centerX * 2) and the width of the object (objWidth)
        float scale = (centerX * 2 / objWidth) * scaleAR;
        float diffX = objCenterX * scale - centerX;
        float diffY = objCenterY * scale - centerY;

        // Loop through all faces
        for (auto face : faces)
        {
            std::vector<cv::Vec3f> tempPoints;

            // Try to project the vertex
            for (auto vertexId : face.vertexesId)
            {
                cv::Vec3f tempPoint = vertexes[vertexId - 1];

                // First we rotate it
                if (rotated)
                {
                    cv::Mat tempPt = (cv::Mat_<double>(3, 1) << tempPoint[0], tempPoint[1], tempPoint[2]);
                    tempPt = R * tempPt;
                    tempPoint = cv::Vec3f(tempPt.at<double>(0, 0), tempPt.at<double>(1, 0), tempPt.at<double>(2, 0));
                }
                // Scale it to world coordinates and move it to align center of the chessboard
                tempPoint *= scale;
                tempPoint[0] = tempPoint[0] - diffX;
                tempPoint[1] = tempPoint[1] - diffY;
                tempPoint[2] *= (-1);
                tempPoints.push_back(tempPoint);
            }
            // Get the projection
            std::vector<cv::Point2f> arPoints;
            cv::projectPoints(tempPoints, rvec, tvec, cameraMatrix, distCoeffs, arPoints);
            // Convert the point to draw the faces,
            std::vector<cv::Point2i> polyPoints;
            for (auto point : arPoints)
            {
                cv::Point2i temp;
                temp.x = (int)point.x;
                temp.y = (int)point.y;
                polyPoints.push_back(temp);
            }
            cv::fillConvexPoly(dst, polyPoints, face.color);
            // Draw the lines to show the boundary if not colored
            if (!colored)
            {
                cv::polylines(dst, polyPoints, true, cv::Scalar(0, 0, 0));
            }
        }
        return 0;
    }
};

#endif
