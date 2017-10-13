#include "image_warping.h"
#include "cl_common.h"

namespace cl {
    namespace camera {

        using namespace std;
        using namespace cv;


        Point2f Warping::get_center(Size img_size) {
            Point2f center;

            if (img_size.width % 2 == 0)
                center.x = float(img_size.width / 2) - 0.5f;
            else
                center.x = ceil(img_size.width / 2);

            if (img_size.height % 2 == 0)
                center.y = float(img_size.height / 2) - 0.5f;
            else
                center.y = ceil(img_size.height / 2);

            return center;
        }


        Point2f Warping::get_rotation(Point2f orig_pt, float angle) {
            Point2f new_loc;
            float orig_theta = atan2(orig_pt.y, orig_pt.x);
            float orig_r = sqrt(pow(orig_pt.y, 2) + pow(orig_pt.x, 2));
            new_loc.x = orig_r * cos(orig_theta - angle * M_PI / 180);
            new_loc.y = orig_r * sin(orig_theta - angle * M_PI / 180);

            return new_loc;
        }



        Point2f Warping::circle_point2shperical_point(Point2f sphere_pt, Camera *camera) {

            Fisheye* fisheye = dynamic_cast<Fisheye*>(camera);

            Vec3d euler = Vec3d(fisheye->euler_x_, fisheye->euler_y_, fisheye->euler_z_);
            Vec3d translation = Vec3d(fisheye->tx_, fisheye->ty_, fisheye->tz_);


            Vec3d _xyz;
            double longitude = 2.0 * M_PI * (sphere_pt.x); // -pi to pi
            double latitude = M_PI * (sphere_pt.y);    // -pi/2 to pi/2


            // Vector in 3D space
            _xyz[0] = cos(latitude) * sin(longitude);
            _xyz[1] = cos(latitude) * cos(longitude);
            _xyz[2] = sin(latitude);

            //unit sphere
            Vec3d xyz;

            xyz[0] = cos(euler[2]) * cos(euler[1]) * _xyz[0] +
                     (-sin(euler[2]) * cos(euler[0]) + cos(euler[2]) * sin(euler[1]) * sin(euler[0])) * _xyz[1] +
                     (sin(euler[2]) * sin(euler[0]) + cos(euler[2]) * sin(euler[1]) * cos(euler[0])) * _xyz[2] +
                     translation[0];

            xyz[1] = sin(euler[2]) * cos(euler[1]) * _xyz[0] +
                     (cos(euler[2]) * cos(euler[0]) + sin(euler[2]) * sin(euler[1]) * sin(euler[0])) * _xyz[1] +
                     (-cos(euler[2]) * sin(euler[0]) + sin(euler[2]) * sin(euler[1]) * cos(euler[0])) * _xyz[2] +
                     translation[1];

            xyz[2] = -sin(euler[1]) * _xyz[0] +
                     cos(euler[1]) * sin(euler[0]) * _xyz[1] +
                     cos(euler[1]) * cos(euler[0]) * _xyz[2] + translation[2];


            if (xyz[1] < 0) {
                return Point2f(-INT_MAX, -INT_MAX);
            }

            double x = xyz[0] / xyz[1], y = xyz[2] / xyz[1];
            double r = sqrt(x * x + y * y);
            double theta = atan(r);

            double theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2, theta8 = theta4 * theta4;
            double theta_d = theta * (1 + fisheye->k1_ * theta2 + fisheye->k2_ * theta4 + fisheye->k3_ * theta6 +
                                      fisheye->k4_ * theta8);

            double scale = (r == 0) ? 1.0 : theta_d / r;
            double u = fisheye->fu_ * x * scale;
            double v = fisheye->fv_ * y * scale;
            return Point2f(u, v);

        }


        void FisheyeWarping::init(Camera *camera) {

            Fisheye* fisheye = dynamic_cast<Fisheye*>(camera);

            Size output_size(fisheye->output_width_, fisheye->output_height_);
            Size input_size(fisheye->input_width_, fisheye->input_height_);


            Mat skew_mapx(output_size, CV_32FC1);
            Mat skew_mapy(output_size, CV_32FC1);
            if (map_x_.empty() || map_x_.size() != output_size || map_x_.type() != CV_32FC1) {
                map_x_.create(output_size, CV_32FC1);
            }

            if (map_y_.empty() || map_y_.size() != output_size || map_y_.type() != CV_32FC1) {
                map_y_.create(output_size, CV_32FC1);
            }


            Point2f src_center(fisheye->center_x_, fisheye->center_y_);
            Point2f dst_center = get_center(output_size);


#pragma omp parallel for num_threads(CL_NUM_THREADS)
            for (int x = 0; x < output_size.width; x++)
            {
                for (int y = 0; y < output_size.height; y++)
                {
                    double x_ = x;
                    double y_ = y;

                    Point2f sphere_Pt = {
                            float(x - dst_center.x) / float(output_size.height * 2),
                            float(y - dst_center.y) / float(output_size.height)
                    };
                    Point2f fish_Pt = circle_point2shperical_point(sphere_Pt, fisheye);
                    Point2f fish_rotated = get_rotation(fish_Pt, fisheye->rotate_angle_);
                    map_x_.at<float>(y, x) = fish_rotated.x + src_center.x;
                    map_y_.at<float>(y, x) = fish_rotated.y + src_center.y;
                }
            }
        }


        void FisheyeWarping::init(Camera *camera, const Rect &mask) {

            Fisheye* fisheye = dynamic_cast<Fisheye*>(camera);
            Size output_size(fisheye->output_width_, fisheye->output_height_);

            Mat skew_mapx(output_size, CV_32FC1);
            Mat skew_mapy(output_size, CV_32FC1);
            if (map_x_.empty() || map_x_.size() != mask.size() || map_x_.type() != CV_32FC1) {
                map_x_.create(mask.size(), CV_32FC1);
            }

            if (map_y_.empty() || map_y_.size() != mask.size() || map_y_.type() != CV_32FC1) {
                map_y_.create(mask.size(), CV_32FC1);
            }


            Point2f src_center(fisheye->center_x_, fisheye->center_y_);
            Point2f dst_center = get_center(output_size);

#pragma omp parallel for num_threads(CL_NUM_THREADS)
            for (int x = 0; x < mask.width; x++) {
                for (int y = 0; y < mask.height; y++) {

                    double x_ = x;
                    double y_ = y;

                    Point2f sphere_Pt =
                    {
                        float(x_ + mask.tl().x - dst_center.x) / float(fisheye->pano_width_),
                        float(y_ + mask.tl().y - dst_center.y) / float(fisheye->pano_width_ / 2)
                    };
                    Point2f fish_Pt = circle_point2shperical_point(sphere_Pt, fisheye);
                    Point2f fish_rotated = get_rotation(fish_Pt, fisheye->rotate_angle_);

                    map_x_.at<float>(y, x) = fish_rotated.x + src_center.x;
                    map_y_.at<float>(y, x) = fish_rotated.y + src_center.y;
                }
            }
        }


        void FisheyeWarping::undistort_image(const Mat &src, Mat &dst, int type) {
            if (type == CV_8UC3) {
                cv::remap(src, dst, map_x_, map_y_, CV_INTER_CUBIC, BORDER_CONSTANT, Scalar(0, 0, 0));//cv_8uc3
            } else if (type == CV_8UC4) {
                Mat temp_bgr, temp_bgra;
                remap(src, temp_bgr, map_x_, map_y_, CV_INTER_CUBIC, BORDER_CONSTANT, Scalar(0, 0, 0));
                cvtColor(temp_bgr, dst, CV_BGR2BGRA); //cv_8uc4
            } else {
                Mat temp_bgr, temp_bgra;
                remap(src, temp_bgr, map_x_, map_y_, CV_INTER_CUBIC, BORDER_CONSTANT, Scalar(0, 0, 0));
                cvtColor(temp_bgr, temp_bgra, CV_BGR2BGRA); //cv_32fc4
                temp_bgra.convertTo(dst, CV_32FC4);
            }
        }








    }
}