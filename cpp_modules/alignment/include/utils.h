//
// Created by root on 17-9-25.
//

#ifndef CL_FACE_UTILS_H
#define CL_FACE_UTILS_H

#include <opencv2/opencv.hpp>
#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <fstream>
#include <numeric>
#include <utility>

class BoundingBox{
public:
    double start_x;
    double start_y;
    double width;
    double height;
    double centroid_x;
    double centroid_y;

    BoundingBox(){
        start_x = 0;
        start_y = 0;
        width = 0;
        height = 0;
        centroid_x = 0;
        centroid_y = 0;
    };

    BoundingBox(int _x, int _y, int _width, int _height){
        start_x = _x;
        start_y = _y;
        width = _width;
        height = _height;
        centroid_x = (start_x + width/2);
        centroid_y = (start_y + height/2);
    };
};

cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> >& shapes,
                              const std::vector<BoundingBox>& bounding_box);

void GetShapeResidual(const std::vector<cv::Mat_<double> >& ground_truth_shapes,
                      const std::vector<cv::Mat_<double> >& current_shapes,
                      const std::vector<BoundingBox>& bounding_boxs,
                      const cv::Mat_<double>& mean_shape,
                      std::vector<cv::Mat_<double> >& shape_residuals);

cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
cv::Mat_<double> ReProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
void SimilarityTransform(const cv::Mat_<double>& shape1, const cv::Mat_<double>& shape2,
                         cv::Mat_<double>& rotation,double& scale);
double calculate_covariance(const std::vector<double>& v_1,
                            const std::vector<double>& v_2);
void LoadData(std::string filepath,
              std::vector<cv::Mat_<uchar> >& images,
              std::vector<cv::Mat_<double> >& ground_truth_shapes,
              std::vector<BoundingBox> & bounding_box);
void LoadDataAdjust(std::string filepath,
                    std::vector<cv::Mat_<uchar> >& images,
                    std::vector<cv::Mat_<double> >& ground_truth_shapes,
                    std::vector<BoundingBox> & bounding_box);
void LoadOpencvBbxData(std::string filepath,
                       std::vector<cv::Mat_<uchar> >& images,
                       std::vector<cv::Mat_<double> >& ground_truth_shapes,
                       std::vector<BoundingBox> & bounding_boxs
);
void LoadCofwTrainData(std::vector<cv::Mat_<uchar> >& images,
                       std::vector<cv::Mat_<double> >& ground_truth_shapes,
                       std::vector<BoundingBox>& bounding_boxs);
void LoadCofwTestData(std::vector<cv::Mat_<uchar> >& images,
                      std::vector<cv::Mat_<double> >& ground_truth_shapes,
                      std::vector<BoundingBox>& bounding_boxs);

BoundingBox CalculateBoundingBox(cv::Mat_<double>& shape);
cv::Mat_<double> LoadGroundTruthShape(std::string& filename);
void adjustImage(cv::Mat_<uchar>& img,
                 cv::Mat_<double>& ground_truth_shape,
                 BoundingBox& bounding_box);

void  TrainModel(std::vector<std::string> trainDataName);
double TestModel(std::vector<std::string> testDataName);
int FaceDetectionAndAlignment(const char* inputname);
void ReadGlobalParamFromFile(std::string path);
double CalculateError(const cv::Mat_<double>& ground_truth_shape, const cv::Mat_<double>& predicted_shape);



#endif //CL_FACE_UTILS_H
