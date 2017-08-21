//
// Created by cl on 17-8-16.
//

#ifndef CL_FACE_DETECTIOR_H
#define CL_FACE_DETECTIOR_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "cl_common.h"

namespace cl {
    namespace fd{
        class Detector{
        public:
            Detector(){};
            virtual ~Detector() {}
            virtual bool detect(cv::Mat& src, std::vector<cl::FaceInfo>& faces, bool draw = false) = 0;
            virtual bool load_model(const std::string& model) = 0;
            virtual bool empty() const = 0;

        DISABLE_COPY_AND_ASSIGN(Detector);
        };
    }
}



#endif //CL_FACE_DETECTIOR_H
