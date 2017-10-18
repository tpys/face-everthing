//
// Created by root on 17-9-25.
//

#ifndef CL_FACE_FACE_POSE_H
#define CL_FACE_FACE_POSE_H

#include "faceParameters.hpp"
#include "faceExtractor.hpp"
#include "cl_common.h"
#include "cl_alignment.h"



namespace op{
    namespace fa{

        class OPFace:public cl::fa::Alignment{
        public:
            OPFace(int input_width = 320, int input_height = 320, int gpu_id = 0):
                    input_width_(input_width), input_height_(input_height), gpu_id_(gpu_id) {}
            virtual ~OPFace(){}

            bool detect(const cv::Mat& src,
                        const std::vector<cl::FaceBox>& windows,
                        std::vector<cl::FaceLandmark>& result){

                if(windows.size() <= 0){
                    return false;
                }

                std::vector<Rectangle<float>> face_rectagenles(windows.size());
                for(size_t i = 0; i < windows.size(); ++i){
                    face_rectagenles[i] = Rectangle<float>(windows[i].bbox_.x,
                                                    windows[i].bbox_.y,
                                                    windows[i].bbox_.width,
                                                    windows[i].bbox_.height);
                }

                extractor_->forwardPass(face_rectagenles, src, 1.f);

                auto keypoints = extractor_->getFaceKeypoints();
                const int number_person = keypoints.getSize(0);
                const int number_keypoints = keypoints.getSize(1);
                const int dim = keypoints.getSize(2);
                result.resize(number_person);

                for(int person = 0; person < number_person; ++person){
                    for(int part = 0; part < number_keypoints; ++part){
                        result[person].points_.resize(number_keypoints);
                        result[person].scores_.resize(number_keypoints);
                        const auto face_idx = (person * number_keypoints + part) * dim;
                        result[person].points_[part] = cv::Point2d(keypoints[face_idx], keypoints[face_idx+1]);
                        result[person].scores_[part] = keypoints[face_idx + 2];
                    }
                }
                return true;
            }


            bool load_model(const std::vector<std::string>& init_nets,
                            const std::vector<std::string>& predict_nets){
                assert((init_nets.size() > 0) && (predict_nets.size() > 0));
                pairs_ = FACE_PAIRS_RENDER;
                extractor_ = std::make_shared<FaceExtractor>(Point<int>(input_width_, input_height_),
                                                             Point<int>(input_width_, input_height_),
                                                             init_nets[0],
                                                             predict_nets[0],
                                                             gpu_id_);

                extractor_->initializationOnThread();
                return true;
            }

        private:
            std::shared_ptr<FaceExtractor> extractor_;
            std::vector<unsigned> pairs_;
            int input_width_;
            int input_height_;
            int gpu_id_;



        DISABLE_COPY_AND_ASSIGN(OPFace);
        };
    }
}





#endif //CL_FACE_FACE_POSE_H
