//
// Created by root on 17-9-26.
//

#ifndef CL_FACE_FLOW_TRACKER_H
#define CL_FACE_FLOW_TRACKER_H

#include "cl_common.h"
#include "cl_alignment.h"
#include "PDM.h"

#include <memory>

namespace cl{
    namespace ft{

        class Tracker{
        public:
            Tracker(const string& pdm_model,
                    const float render_threshold = 0.4,
                    cl::fa::Alignment* alignment = nullptr);
            virtual ~Tracker(){}

            bool tracking(const cv::Mat& bgr_image, const cv::Rect& bouding_box = {});
            void reset();

            cv::Rect get_bbox() { return face_rect_;}

            void draw(cv::Mat& display_img);

            struct Param{
                float face_template_scale_;
                bool use_face_template_;
            };

        private:
            void match_template(const cv::Mat& bgr_image);
            void update_template(const cv::Mat& bgr_image);
            bool check_landmarks(const cl::FaceLandmark& landmarks);

            cv::Rect compute_bbox() const {
                return cv::boundingRect(result_[0].points_);
            }


        public:
            int failures_;
            bool tracking_success_;
        private:
            cv::Mat face_template_;
            cv::Rect face_rect_;
            bool initialised_;
            float render_threshold_;
            Param param_;
            LandmarkDetector::PDM pdm_;
            cv::Mat_<double> params_local_;
            cv::Vec6d params_global_;
            std::shared_ptr<cl::fa::Alignment> face_alignment_;
            vector<cl::FaceLandmark> result_;
            const int DrawShiftbits = 0;
            const int DrawMultiplier = 2;

        DISABLE_COPY_AND_ASSIGN(Tracker);
        };

    }
}












#endif //CL_FACE_FLOW_TRACKER_H
