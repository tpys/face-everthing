//
// Created by root on 17-8-29.
//

#include "sphereface.h"

using namespace std;
using namespace cv;

namespace sf{
    namespace fr{
        SphereFace::SphereFace(int batch_size,
                               int channels,
                               int height,
                               int width,
                               bool do_flip):batch_size_(batch_size), channels_(channels), height_(height), width_(width),do_flip_(do_flip)
        {
            if(do_flip_)
            {
                batch_size_ = 2*batch_size;
            }
        }

        int SphereFace::extract_feature_or_identify(const std::vector<cv::Mat>& face_images,
                                                    const std::vector<cv::Rect>& face_windows,
                                                    const std::vector<std::vector<cv::Point2f>>& face_landmarks,
                                                    std::vector<std::vector<float>>& features) {
            assert(face_windows.size() == face_landmarks.size());
            vector<float> input_blob;
            for(size_t i = 0; i < face_images.size(); ++i){
                if(face_windows.size() == 0){
                    do_ensemble_ = false;
                }
                this->preprocess(face_images[i], face_landmarks[i], input_blob);
            }
            sphere20_->forward_pass(input_blob.data());
            int src_image_num = face_images.size();
            sphere20_->extract_feature(src_image_num, features);
        }

        vector<float> SphereFace::extract_feature(cv::Mat& aligned_face){
            vector<float> input_blob;
            preprocess(aligned_face, {}, input_blob);
            sphere20_->forward_pass(input_blob.data());
            return sphere20_->extract_feature();

        }

        void SphereFace::preprocess(const cv::Mat& src,
                                   const std::vector<cv::Point2f>& landmarks,
                                   std::vector<float>&blob_data){

            //resize
            Mat image = src.clone();
            image.convertTo(image, CV_32FC3, 1.0/128, -127.5/128);

            //NHWC to NCHW
            vector<cv::Mat> channels(3);
            cv::split(image, channels);
            for (auto &c : channels) {
                blob_data.insert(blob_data.end(), (float *)c.datastart, (float *)c.dataend);
            }


            if(do_flip_){
                Mat image_flip;
                flip(image, image_flip, 0);
                vector<cv::Mat> channels_flip(3);
                cv::split(image_flip, channels_flip);
                for (auto &c : channels_flip) {
                    blob_data.insert(blob_data.end(), (float *)c.datastart, (float *)c.dataend);
                }
            }
        }


        bool SphereFace::load_model(const std::vector<std::string>& init_nets,
                                    const std::vector<std::string>& predict_nets){

            sphere20_ = std::make_shared<cl::NetCaffe>(init_nets[0], predict_nets[0], "fc5");
            return sphere20_->init(batch_size_, channels_, height_, width_);
        }


    }
}

