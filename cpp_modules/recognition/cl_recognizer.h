//
// Created by root on 17-8-29.
//

#ifndef CL_FACE_RECOGNITION_H
#define CL_FACE_RECOGNITION_H

#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <fstream>

#include "cl_common.h"

namespace cl{
    namespace fr{
        class Recognizer{
        public:
            Recognizer(){};
            virtual ~Recognizer() {}

            virtual int extract_feature_or_identify(const std::vector<cv::Mat>& face_images,
                                   const std::vector<cv::Rect>& face_windows,
                                   const std::vector<std::vector<cv::Point2f>>& face_landmarks,
                                   std::vector<std::vector<float>>& features) = 0;

            virtual std::vector<float> extract_feature(cv::Mat& aligned_face) = 0;


            virtual bool load_model(const std::vector<std::string>& init_nets,
                                    const std::vector<std::string>& predict_nets) = 0;

        private:
            DISABLE_COPY_AND_ASSIGN(Recognizer);
        };


        class FaceClassifier{
        public:
            FaceClassifier():class_num_(0), feature_dim_(1024), score_thresh_(0.60) {}
            FaceClassifier(int class_num, int feature_dim, float score_thresh):class_num_(class_num),
                                                                               feature_dim_(feature_dim),
                                                                               score_thresh_(score_thresh) {}


            bool load_mean(const std::string& mean_file){
                std::ifstream file(mean_file);
                if(!file.is_open()) return false;
                feature_mean_.create(1, feature_dim_, CV_32FC1);
                for(size_t i = 0; i < feature_dim_; ++i){
                    file >> feature_mean_.at<float>(i);
                }
                return true;
            }

            int identify(const std::vector<float>& feature) {
                if(class_num_ == 0) {
                    return -1;
                }
                else{
                    return first_guess(feature);
                }
            }

            std::vector<int>& neighbours(const std::vector<float>& feature);
            std::string get_name(int id) {
                if (id < 0){
                    return "unknown";
                } else{
                    return known_people_[id];
                }
            }

            void add_person(const std::string& name, const std::vector<float>& feature) {
                known_people_[class_num_] = name;
                cv::Mat feature_mat = cv::Mat(1, feature_dim_, CV_32FC1, (void*)feature.data());
                cv::Mat feature_normalize;
                cv::normalize(feature_mat - feature_mean_, feature_normalize);
                features_.push_back(feature_normalize);
                ++class_num_;
            }

        private:
            int first_guess(const std::vector<float>& feature) {
                std::vector<float> scores(class_num_, 0);
                for(int i = 0; i < class_num_; ++i){
                    cv::Mat feature_mat = cv::Mat(1, feature_dim_, CV_32FC1, (void*)feature.data());
                    cv::Mat feature_normalize;
                    cv::normalize(feature_mat - feature_mean_, feature_normalize);
                    scores[i] = feature_normalize.dot(features_[i]);
                }
                auto iter = std::max_element(begin(scores), end(scores));
                int max_id =  std::distance(scores.begin(), iter);
                float max_score = scores[max_id];
//                std::cout << "max_score: " << max_score << " max_id: " << max_id << std::endl;
                if(max_score < score_thresh_){
                    return -1;
                }
                return max_id;
            }


        private:
            std::map<int, std::string> known_people_;
            std::vector<cv::Mat> features_;
            cv::Mat feature_mean_;


            int class_num_;
            int feature_dim_;
            float score_thresh_;

        };

    }
}

#endif //CL_FACE_RECOGNITION_H
