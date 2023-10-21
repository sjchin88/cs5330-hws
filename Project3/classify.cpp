/*
  Class Name    : CS5330 Pattern Recognition and Computer Vision
  Session       : Fall 2023 (Seattle)
  Name          : Shiang Jin Chin
  Last Update   : 10/20/2023
  Description   : Contain all required methods for classification for project 3

*/

#include "classify.h"
/**
 * Helper method to compute the distance between target's features and object database features
 * based on normalized euclidean distance
 * input : targetFeatures: features of target object
 * input : vector<ObjectStruct> &objectLists - list of object data from the database
 * input : vector<double> stdSquares - precomputed standard deviation for each features based on the database
 * output: vector<ResultStruct> &resultList - list of distance result for each object in the database
 */
int computeEucDist(vector<double> &targetFeatures, vector<ObjectStruct> &objectLists, vector<double> stdSquares, vector<ResultStruct> &resultList)
{
    try
    {
        // Next we compute the normalized euclidean distance between target's features and each object
        // and store it in result List
        for (int i = 0; i < objectLists.size(); i++)
        {
            double totalDist = 0.0;
            for (int j = F_PERCENT_FILLED; j <= F_HU_7TH_MOMENT; j++)
            {
                double diff = targetFeatures[j] - objectLists[i].features[j];
                double dist = diff * diff / stdSquares[j];
                totalDist += dist;
            }
            resultList.push_back(ResultStruct(objectLists[i].objName, totalDist));
        }
    }
    catch (exception)
    {
        return (-1);
    }
    return 0;
}

/**
 * Read the database and precompute the statistic for reuse
 * input : csvFilePath of the objectDB
 * output: vector<ObjectStruct> &objectLists of all object extracted from csvFilePath
 * output: StatStruct &commonStat for the standard deviation squares and average distance to be used by all
 */
int computeDB(string &csvFilePath, vector<ObjectStruct> &objectLists, StatStruct &commonStat)
{
    try
    {
        // extract obj names and data from csv file
        vector<char *> objNames;
        vector<vector<float>> objData;
        char csvPath[256];
        strcpy_s(csvPath, csvFilePath.c_str());
        if (read_image_data_csv(csvPath, objNames, objData) != 0)
        {
            cout << "error reading object database csv file " << endl;
            return (-1);
        };

        // update it to the list
        for (int i = 0; i < objNames.size(); i++)
        {
            ObjectStruct tempObj(objNames[i], objData[i]);
            objectLists.push_back(tempObj);
        }

        // First get the standard deviations for each features in the database
        vector<double> stdSquares;
        // In the object data, only features from F_PERCENT_FILLED to F_HU_7TH_MOMENTS
        // are used in distance calculation
        for (int i = F_PERCENT_FILLED; i <= F_HU_7TH_MOMENT; i++)
        {
            // Get the mean
            int n = objData.size();
            double total = 0.0;
            for (int j = 0; j < n; j++)
            {
                total += objData[j][i];
            }
            double mean = total / n;

            // Get the sum of error square
            double sumErrSquare = 0.0;
            for (int j = 0; j < n; j++)
            {
                double err = objData[j][i] - mean;
                sumErrSquare += err * err;
            }

            // Get the std square and pushback to the vector
            double stdSquare = sumErrSquare / n;
            stdSquares.push_back(stdSquare);
        }

        // Try to get the avg distance, based on first object
        vector<ObjectStruct> tempList;
        ObjectStruct target = objectLists[0];
        for (int i = 1; i < objectLists.size(); i++)
        {
            if (target.objName != objectLists[i].objName)
            {
                tempList.push_back(objectLists[i]);
            }
        }
        vector<double> targetFeatures;
        for (auto feature : target.features)
        {
            targetFeatures.push_back(feature);
        }
        vector<ResultStruct> tempResult;
        computeEucDist(targetFeatures, tempList, stdSquares, tempResult);

        // Sum up and get the avg distance
        double totalDiff = 0;
        for (auto result : tempResult)
        {
            totalDiff += result.distance;
        }
        double avgDiff = totalDiff / tempResult.size();
        commonStat = StatStruct(stdSquares, avgDiff);
    }
    catch (exception)
    {
        return (-1);
    }
    return 0;
}

/**
 * Read the database and find the closest object based on normalize euclidean distance
 * input : vector of computed features
 * input : vector<ObjectStruct> &objectLists
 * input : StatStruct &commonStat for the standard deviation squares and average distance to be used by all
 * Output: result name of the closest object
 *
 */
int classifyObject(vector<double> &features, vector<ObjectStruct> &objectLists, StatStruct &commonStat, string &result, int k)
{
    try
    {
        if (k <= 0)
        {
            cout << "invalid k value, need to be > 0 " << endl;
            return (-1);
        }

        vector<ResultStruct> resultList;
        if (computeEucDist(features, objectLists, commonStat.stdSquares, resultList) != 0)
        {
            cout << "error computing euclidean distance " << endl;
            return (-1);
        }
        // Sort the result with lowest distance first
        sort(resultList.begin(), resultList.end());

        // if the lowest distance is larger than the avg distance in the DB
        // likely the obj is unknown
        if (resultList[0].distance > commonStat.avgDist)
        {
            result = string("unknown");
            return (0);
        }
        // Debug code, print the distance for each objects
        /*for (int i = 0; i < resultList.size(); i++)
        {
            ResultStruct res = resultList[i];
            std::cout << res.objName << " has distance of " << res.distance << endl;
        }*/
        // if k == 1, return first object name
        if (k == 1)
        {
            result = string(resultList[0].objName);
            return (0);
        }

        // Else, initialize a map to store the counts
        std::map<string, int> obj2Cnt;
        int limit = std::min<int>(resultList.size(), k);
        for (int i = 0; i < limit; i++)
        {
            // printf("object %i is %s", i, resultList[i].objName);
            if (obj2Cnt.find(resultList[i].objName) == obj2Cnt.end())
            {
                obj2Cnt[resultList[i].objName] = 1;
            }
            else
            {
                obj2Cnt[resultList[i].objName] += 1;
            }
        }

        string maxObj = string(resultList[0].objName);
        int maxCnt = obj2Cnt[maxObj];
        for (const auto &object : obj2Cnt)
        {
            if (object.second > maxCnt)
            {
                maxCnt = object.second;
                maxObj = string(object.first);
            }
        }
        result = maxObj;
    }
    catch (exception)
    {
        return (-1);
    }
    return 0;
}