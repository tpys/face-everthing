#ifndef CL_FACE_UTILS_CAMERA_H
#define CL_FACE_UTILS_CAMERA_H

#include <opencv2/opencv.hpp>

namespace cl {
    namespace camera {


        struct Camera{
            Camera() {}
            virtual ~Camera(){}
        };


        struct Fisheye: public Camera{

            Fisheye(int pano_width = 1920,
                   int input_width = 1296,
                   int input_height = 992,
                   int output_width = 960,
                   int output_height = 960,
                   int crop_width = 960,
                   int crop_height = 960,
                   float fu = 3.2709526837916417e+02,
                   float fv = 3.3243670491075636e+02,
                   float k1 = -7.7253987204226396e-02,
                   float k2 = 2.0669709869175927e-02,
                   float k3 = -8.3624290157720479e-03,
                   float k4 = 2.6240108850240388e-03,
                   float center_x = 6.4750000000000000e+02,
                   float center_y = 4.9550000000000000e+02,
                   float euler_x = 0,
                   float euler_y = 0,
                   float euler_z = 0,
                   float tx = 0,
                   float ty = 0,
                   float tz = 0,
                   float rotate_angle = -90):pano_width_(pano_width),
                                             input_width_(input_width), input_height_(input_height),
                                             output_width_(output_width), output_height_(output_height),
                                             crop_width_(crop_width), crop_height_(crop_height),
                                             fu_(fu), fv_(fv), k1_(k1), k2_(k2), k3_(k3), k4_(k4),
                                             center_x_(center_x), center_y_(center_y),
                                             euler_x_(euler_x), euler_y_(euler_y), euler_z_(euler_z),
                                             tx_(tx), ty_(ty), tz_(tz),
                                             rotate_angle_(rotate_angle){
            }

            virtual ~Fisheye() {};

            int pano_width_;
            int input_width_;
            int input_height_;
            int output_width_;
            int output_height_;
            int crop_width_;
            int crop_height_;

            float fu_;
            float fv_;
            float k1_;
            float k2_;
            float k3_;
            float k4_;

            float center_x_;
            float center_y_;

            float euler_x_;
            float euler_y_;
            float euler_z_;
            float tx_;
            float ty_;
            float tz_;

            float rotate_angle_ = -90;
            cv::Mat projection_ = cv::Mat();
        };









    }
}
#endif