//
// Created by root on 17-9-25.
//

#ifndef CL_FACE_FPS3000_ALIGNMENT_H
#define CL_FACE_FPS3000_ALIGNMENT_H


#include "LBFRegressor.h"
#include "LBF.h"

namespace lbf{
    namespace fa{

        class LBFAlignment:public cl::fa::Alignment{
        public:
            LBFAlignment(int iteration_num = 20):iteration_num_(iteration_num){}
            virtual ~LBFAlignment(){}

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

                Params params_default;
                regressor_ = std::make_shared<LBFRegressor>(params_default);
                regressor_->Load(init_nets[0], predict_nets[0]);
                return true;
            }

        private:
            const int LandmarkNum = 68;
            int iteration_num_;
            std::shared_ptr<LBFRegressor> regressor_;
            DISABLE_COPY_AND_ASSIGN(LBFAlignment);
        };
    }
}






#endif //CL_FACE_FPS3000_ALIGNMENT_H
