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
            virtual bool detect(const cv::Mat& src, std::vector<cl::FaceBox>& faces) = 0;
            virtual bool load_model(const std::vector<std::string>& init_nets,
                                    const std::vector<std::string>& predict_nets) = 0;


            cv::Rect get_biggest_rect(const std::vector<cl::FaceBox>& faces, int src_width, int src_height) {
                assert(faces.size() > 0);
                int idx = 0;
                float dist_max = 0;
                if (faces.size() == 1) {
                    return faces[0].bbox_;
                } else {
                    for(size_t i = 0; i < faces.size(); ++i) {
                        float area = faces[i].bbox_.area();
                        cv::Point2f offset(faces[i].bbox_.x + faces[i].bbox_.width/2.0f - src_width/2.0f,
                                           faces[i].bbox_.y + faces[i].bbox_.height/2.0f - src_height/2.0f);
                        float dist_squared = area - (offset.x * offset.x + offset.y * offset.y);
                        if (dist_max < dist_squared){
                            dist_max = dist_squared;
                            idx = i;
                        }
                    }
                    return faces[idx].bbox_;
                }
            }



        private:
        DISABLE_COPY_AND_ASSIGN(Detector);

        };
    }
}



#endif //CL_FACE_DETECTIOR_H
