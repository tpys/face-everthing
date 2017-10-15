#ifndef CL_FACE_UTILS_WRAPER_H
#define CL_FACE_UTILS_WRAPER_H

#include "camera_model.h"

namespace cl {
    namespace camera {


        class Warping {
        public:
            virtual ~Warping() {}
            virtual void init(Camera *camera) = 0;
            virtual void init(Camera *camera, const cv::Rect &mask) = 0;
            virtual void undistort_image(const cv::Mat &src, cv::Mat &dst, int type = CV_8UC3) = 0;

            cv::Point2f get_center(cv::Size img_size);
            cv::Point2f get_rotation(cv::Point2f orig_pt, float angle);
            cv::Point2f circle_point2shperical_point(cv::Point2f sphere_pt, Camera *camera);
        };


        class FisheyeWarping : public Warping {
        public:
            FisheyeWarping() {}
            void init(Camera *camera);
            void init(Camera *camera, const cv::Rect &mask);
            void undistort_image(const cv::Mat &src, cv::Mat &dst, int type = CV_8UC3);

        private:
            cv::Mat map_x_;
            cv::Mat map_y_;
        };



    }
}

#endif