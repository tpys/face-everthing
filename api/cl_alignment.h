//
// Created by root on 17-8-18.
//
#ifndef CL_FACE_ALIGNMENT_H_H
#define CL_FACE_ALIGNMENT_H_H


#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "cl_common.h"
#include "cl_detector.h"

namespace cl {
    namespace fa{
        class Alignment{
        public:
            Alignment(){};
            virtual ~Alignment() {}
            virtual bool detect(cv::Mat& src, cl::FaceInfo& face, bool draw = false) = 0;
            virtual void align(const cv::Mat& src, cl::FaceInfo& face, cv::Mat& dst) = 0;
            virtual bool load_model(const std::string& detection_model) = 0;
            virtual bool empty() const = 0;
            virtual void set_detector(cl::fd::Detector* detector) = 0;


        DISABLE_COPY_AND_ASSIGN(Alignment);
        };
    }
}




#endif //CL_FACE_ALIGNMENT_H_H
