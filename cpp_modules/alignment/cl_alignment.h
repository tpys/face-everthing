//
// Created by root on 17-8-18.
//
#ifndef CL_FACE_ALIGNMENT_H_H
#define CL_FACE_ALIGNMENT_H_H


#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "cl_common.h"
#include "cl_detector.h"

namespace cl {
    namespace fa{

        static cv::Rect_<double> landmark2rect(const std::vector<cv::Point2f>& landmark, double scale = 1.5, double shift_scale = 0) {
            auto landmark_rect = cv::boundingRect(landmark);
            cv::Rect_<double> face_rect = {
                    std::max<double>(0, landmark_rect.x - scale*landmark_rect.width/2),
                    std::max<double>(0, landmark_rect.y - (scale + shift_scale)*landmark_rect.width/2),
                    (1 + scale)*landmark_rect.width,
                    (1 + scale)*landmark_rect.width,
            };
            return face_rect;
        }

        static cv::Rect_<double> landmark2rect(const cl::FaceLandmark& landmark, double scale = 1.5, double shift_scale = 0) {

            return landmark2rect(landmark.points_, scale, shift_scale);
        }

        static int get_biggest_id(const std::vector<cl::FaceLandmark>& landmarks, int src_width, int src_height) {
            assert(landmarks.size() > 0);
            int idx = 0;
            float dist_max = 0;
            if (landmarks.size() == 1) {
                return 0;
            } else {
                for(size_t i = 0; i < landmarks.size(); ++i) {
                    auto bbox = cv::boundingRect(landmarks[i].points_);
                    float area = bbox.area();
                    cv::Point2f offset(bbox.x + bbox.width/2.0f - src_width/2.0f,
                                       bbox.y + bbox.height/2.0f - src_height/2.0f);
                    float dist_squared = area - (offset.x * offset.x + offset.y * offset.y);
                    if (dist_max < dist_squared){
                        dist_max = dist_squared;
                        idx = i;
                    }
                }
                return idx;
            }
        }


        static void draw(const cl::FaceLandmark& landmarks, cv::Mat& display_img) {
            const int DrawShiftbits = 0;
            const int DrawMultiplier = 2;
            int thickness = (int)std::ceil(3.0* ((double)display_img.cols) / 640.0);
            int thickness_2 = (int)std::ceil(1.0* ((double)display_img.cols) / 640.0);

            for(auto& e: landmarks.points_){
                cv::circle(display_img, cv::Point(e.x, e.y), 1 * DrawMultiplier, cv::Scalar(0, 0, 255), thickness, CV_AA, DrawShiftbits);
                cv::circle(display_img, cv::Point(e.x, e.y), 1 * DrawMultiplier, cv::Scalar(255, 0, 0), thickness_2, CV_AA, DrawShiftbits);

            }
        }

        static void draw(const cl::FaceBox& face_box, const cl::FaceLandmark& landmarks, cv::Mat& display_img) {
            const int DrawShiftbits = 0;
            const int DrawMultiplier = 2;
            int thickness = (int)std::ceil(3.0* ((double)display_img.cols) / 640.0);
            int thickness_2 = (int)std::ceil(1.0* ((double)display_img.cols) / 640.0);

            for(auto& e: landmarks.points_){
                cv::circle(display_img, cv::Point(e.x, e.y), 1 * DrawMultiplier, cv::Scalar(0, 0, 255), thickness, CV_AA, DrawShiftbits);
                cv::circle(display_img, cv::Point(e.x, e.y), 1 * DrawMultiplier, cv::Scalar(255, 0, 0), thickness_2, CV_AA, DrawShiftbits);

            }
            cv::rectangle(display_img, face_box.bbox_, cv::Scalar(255, 0, 0), 2);
        }


        class Alignment{
        public:
            Alignment(){};
            virtual ~Alignment() {}

            virtual bool detect(const cv::Mat& src,
                                const std::vector<cl::FaceBox>& windows,
                                std::vector<cl::FaceLandmark>& result) = 0;

            virtual bool load_model(const std::vector<std::string>& init_nets,
                                    const std::vector<std::string>& predict_nets) = 0;

            virtual void align_face(const cv::Mat& src,
                                    const std::vector<cl::FaceLandmark>& landmark,
                                    int width,
                                    int height, std::vector<cv::Mat>& aligned_faces) {
                //ToDo
            }

            virtual cv::Mat align_face(const cv::Mat& src,
                                       const cl::FaceLandmark& landmark,
                                       int width,
                                       int height) {
                //ToDo
                return {};
            }


        private:
            DISABLE_COPY_AND_ASSIGN(Alignment);
        };
    }
}




#endif //CL_FACE_ALIGNMENT_H_H
