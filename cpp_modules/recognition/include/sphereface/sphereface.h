//
// Created by root on 17-8-29.
//
#ifndef CL_FACE_SPHEREFACE_H
#define CL_FACE_SPHEREFACE_H

#include "cl_recognizer.h"
#include "net_caffe.h"

namespace sf{
    namespace fr{
        class SphereFace: public cl::fr::Recognizer{
        public:
            explicit SphereFace(int batch_size = 1, int channels = 3, int height = 112, int width = 96, bool do_flip = false);
            ~SphereFace(){}

            virtual int extract_feature_or_identify(const std::vector<cv::Mat>& face_images,
                                                    const std::vector<cv::Rect>& face_windows,
                                                    const std::vector<std::vector<cv::Point2f>>& face_landmarks,
                                                    std::vector<std::vector<float>>& features);

            std::vector<float> extract_feature(cv::Mat& aligned_face);

            virtual bool load_model(const std::vector<std::string>& init_nets,
                                    const std::vector<std::string>& predict_nets);

        private:
            void preprocess(const cv::Mat& src,
                           const std::vector<cv::Point2f>& landmarks,
                           std::vector<float>&blob_data);
        private:
            int batch_size_;
            int channels_;
            int height_;
            int width_;
            bool do_flip_;
            bool do_ensemble_;
            std::shared_ptr<cl::NetCaffe> sphere20_;

        DISABLE_COPY_AND_ASSIGN(SphereFace);
        };

    }

}


#endif //CL_FACE_SPHEREFACE_H
