/*
Author: Bi Sai 
Date: 2014/06/18
This program is a reimplementation of algorithms in "Face Alignment by Explicit 
Shape Regression" by Cao et al.
If you find any bugs, please email me: soundsilencebisai-at-gmail-dot-com

Copyright (c) 2014 Bi Sai 
The MIT License (MIT)
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/


#ifndef FACE_ALIGNMENT_ES_H
#define FACE_ALIGNMENT_ES_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <fstream>
#include <numeric>   
#include <utility>
#include <memory>


#include "cl_alignment.h"
#include "cl_common.h"
#include "utils.h"

class Fern{
    private:
        int fern_pixel_num_;
        int landmark_num_;
        cv::Mat_<int> selected_nearest_landmark_index_;
        cv::Mat_<double> threshold_;
        cv::Mat_<int> selected_pixel_index_;
        cv::Mat_<double> selected_pixel_locations_;
        std::vector<cv::Mat_<double> > bin_output_;
    public:
        std::vector<cv::Mat_<double> > Train(const std::vector<std::vector<double> >& candidate_pixel_intensity, 
                                             const cv::Mat_<double>& covariance,
                                             const cv::Mat_<double>& candidate_pixel_locations,
                                             const cv::Mat_<int>& nearest_landmark_index,
                                             const std::vector<cv::Mat_<double> >& regression_targets,
                                             int fern_pixel_num);
        cv::Mat_<double> Predict(const cv::Mat_<uchar>& image,
                                 const cv::Mat_<double>& shape,
                                 const cv::Mat_<double>& rotation,
                                 const BoundingBox& bounding_box,
                                 double scale);
        void Read(std::ifstream& fin);
        void Write(std::ofstream& fout);
};

class FernCascade{
    public:
        std::vector<cv::Mat_<double> > Train(const std::vector<cv::Mat_<uchar> >& images,
                                             const std::vector<cv::Mat_<double> >& current_shapes,
                                             const std::vector<cv::Mat_<double> >& ground_truth_shapes,
                                             const std::vector<BoundingBox> & bounding_box,
                                             const cv::Mat_<double>& mean_shape,
                                             int second_level_num,
                                             int candidate_pixel_num,
                                             int fern_pixel_num,
                                             int curr_level_num,
                                             int first_level_num);  
        cv::Mat_<double> Predict(const cv::Mat_<uchar>& image, 
                                 const BoundingBox& bounding_box, 
                                 const cv::Mat_<double>& mean_shape,
                                 const cv::Mat_<double>& shape);
        void Read(std::ifstream& fin);
        void Write(std::ofstream& fout);
    private:
        std::vector<Fern> ferns_;
        int second_level_num_;
};

class ShapeRegressor{
    public:
        ShapeRegressor(); 
        void Train(const std::vector<cv::Mat_<uchar> >& images, 
                   const std::vector<cv::Mat_<double> >& ground_truth_shapes,
                   const std::vector<BoundingBox>& bounding_box,
                   int first_level_num, int second_level_num,
                   int candidate_pixel_num, int fern_pixel_num,
                   int initial_num);
        cv::Mat_<double> Predict(const cv::Mat_<uchar>& image, const BoundingBox& bounding_box, int initial_num);
        void Read(std::ifstream& fin);
        void Write(std::ofstream& fout);
        void Load(std::string path);
        void Save(std::string path);
    private:
        int first_level_num_;
        int landmark_num_;
        std::vector<FernCascade> fern_cascades_;
        cv::Mat_<double> mean_shape_;
        std::vector<cv::Mat_<double> > training_shapes_;
        std::vector<BoundingBox> bounding_box_;
};

//cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> >& shapes,
//                              const std::vector<BoundingBox>& bounding_box);
//cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
//cv::Mat_<double> ReProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
//void SimilarityTransform(const cv::Mat_<double>& shape1, const cv::Mat_<double>& shape2,
//                         cv::Mat_<double>& rotation,double& scale);
//double calculate_covariance(const std::vector<double>& v_1,
//                            const std::vector<double>& v_2);



namespace es{
    namespace fa{

        class ExplicitShape:public cl::fa::Alignment{
        public:
            ExplicitShape(int iteration_num = 20):iteration_num_(iteration_num){}
            virtual ~ExplicitShape(){}

            bool detect(const cv::Mat& src,
                        const std::vector<cl::FaceBox>& windows,
                        std::vector<cl::FaceLandmark>& result){

                cv::Mat img_gray;
                if (src.channels() != 1)
                    cv::cvtColor(src, img_gray, cv::COLOR_BGR2GRAY);
                else
                    img_gray = src;
                result.resize(windows.size());

//                #pragma omp parallel for num_threads(CL_NUM_THREADS)
                for(size_t i = 0; i < windows.size(); ++i){
                    BoundingBox bbox(windows[i].bbox_.x, windows[i].bbox_.y, windows[i].bbox_.width, windows[i].bbox_.height);
                    cv::Mat_<double> current_shape = regressor_->Predict(img_gray, bbox, iteration_num_);
                    result[i].points_.resize(LandmarkNum);
                    for(int j = 0; j < LandmarkNum; ++j ){
                        result[i].points_[j] = cv::Point2d(current_shape(j,0), current_shape(j,1));
                    }
                }
            }

            bool load_model(const std::vector<std::string>& init_nets,
                            const std::vector<std::string>& predict_nets){
                if(predict_nets.size() <= 0) {
                    return false;
                }
                regressor_ = std::make_shared<ShapeRegressor>();
                regressor_->Load(predict_nets[0]);
                return true;
            }

        private:
            const int LandmarkNum = 29;
            int iteration_num_;
            std::shared_ptr<ShapeRegressor> regressor_;
            DISABLE_COPY_AND_ASSIGN(ExplicitShape);
        };
    }
}

#endif
