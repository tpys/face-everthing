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


#define CL_NUM_THREADS 4

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

    struct FaceInfo
    {
        enum { LEFT_EYE, RIGHT_EYE, NOSE, LEFT_MOUTH, RIGHT_MOUTH  };
        cv::Rect bbox_;
        std::vector<cv::Point2f> landmarks_;
        std::vector<cv::Rect> patchs_;
        cv::Vec6d head_pose_;
        double face_score_;
        double landmark_score_;
    };


    const int TrainWidth = 182;
    const int TrainHeight = 182;
    const int TestWidth = 160;
    const int TestHeight = 160;

}
#endif //CL_FACE_COMMON_H
