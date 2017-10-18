//
// Created by cl on 17-8-16.
//

#ifndef CL_FACE_COMMON_H
#define CL_FACE_COMMON_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <vector>
#include <iterator>
#include <map>

#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname)                              \
private:                                                                       \
  classname(const classname&) = delete;                                        \
  classname& operator=(const classname&) = delete
#endif


#define CL_NUM_THREADS 8
#define USE_CAFFE 1

namespace cl
{
    template<typename Out>
    void split(const std::string &s, char delim, Out result) {
        std::stringstream ss;
        ss.str(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            *(result++) = item;
        }
    }

    static std::vector<std::string> split(const std::string &s, char delim) {
        std::vector<std::string> elems;
        split(s, delim, std::back_inserter(elems));
        return elems;
    }

    struct FaceBox{
        FaceBox(){}
        FaceBox(const cv::Rect& bbox, float score = 0):bbox_(bbox), score_(score) {}
        FaceBox(float x1, float y1, float x2, float y2, float score = 0) {
            bbox_.x = x1;
            bbox_.y = y1;
            bbox_.width = x2 - x1;
            bbox_.height = y2 - y1;
            score_ = score;
        }

        cv::Rect bbox_{0, 0, 0, 0};
        double score_;
    };

    struct FaceLandmark{
        FaceLandmark():points_{}, scores_{} {}
        FaceLandmark(const std::vector<cv::Point2f>& points):points_(points) {}
        FaceLandmark(const std::vector<cv::Point2f>& points,
                     const std::vector<float>& scores):points_(points), scores_(scores) {}

        std::vector<cv::Point2f> points_;
        std::vector<float> scores_;
    };


    struct FaceInfo
    {
        int identity_;
    };


    const int TrainWidth = 182;
    const int TrainHeight = 182;
    const int TestWidth = 160;
    const int TestHeight = 160;

}
#endif //CL_FACE_COMMON_H
