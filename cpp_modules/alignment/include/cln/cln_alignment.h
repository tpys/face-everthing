//
// Created by root on 17-8-18.
//

#ifndef CL_FACE_OPENFACE_ALIGNMENT_H
#define CL_FACE_OPENFACE_ALIGNMENT_H

#include "LandmarkDetectorModel.h"

#include "cl_alignment.h"
#include "cl_common.h"

namespace cln{
    namespace fa{

        class CLNAlignment:public cl::fa::Alignment{
        public:
            CLNAlignment(
                    bool use_face_template = false,
                    bool multi_view = false,
                    double sigma = 1.5,
                    double reg_facetor = 25,
                    double weight_factor = 0,
                    double validation_boundary = -0.6,
                    double cx = 0,
                    double cy = 0,
                    double fx = 0,
                    double fy = 0);

            virtual ~CLNAlignment(){}


            bool detect(const cv::Mat& src,
                        const std::vector<cl::FaceBox>& windows,
                        std::vector<cl::FaceLandmark>& result);

            bool load_model(const std::vector<std::string>& init_nets,
                                    const std::vector<std::string>& predict_nets);

            void set_detector(cl::fd::Detector* detector) { clnf_model_->face_detector.reset(detector); }

            cv::Rect_<double> get_bbx() const {
                return clnf_model_->GetBoundingBox();
            }


        private:
            inline bool is_valid_face(const cv::Vec6d& pose) const;
            void extract_rigid_points(cv::Mat_<double>& source_points, cv::Mat_<double>& destination_points);

        private:
            const double EulerThresh = 20;
            const double SimScale = 0.8;

            double fx_, fy_, cx_, cy_;
            LandmarkDetector::FaceModelParameters model_param_;
            shared_ptr< LandmarkDetector::CLNF> clnf_model_;

            DISABLE_COPY_AND_ASSIGN(CLNAlignment);
        };
    }
}




#endif //CL_FACE_OPENFACE_ALIGNMENT_H
