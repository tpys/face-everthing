//
// Created by root on 17-8-18.
//

#include "cln_alignment.h"
#include "LandmarkDetectorFunc.h"

using namespace std;
using namespace cln::fa;
using namespace LandmarkDetector;

CLNAlignment::CLNAlignment(
        bool use_face_template,
        bool multi_view,
        double sigma,
        double reg_facetor,
        double weight_factor,
        double validation_boundary,
        double cx,
        double cy,
        double fx,
        double fy)
{
    model_param_.use_face_template = use_face_template;
    model_param_.multi_view = multi_view;
    model_param_.sigma = sigma;
    model_param_.reg_factor = reg_facetor;
    model_param_.weight_factor = weight_factor;
    model_param_.validation_boundary = validation_boundary;
    cx_ = cx;
    cy_ = cy;
    fx_ = fx;
    fy_ = fy;
}


bool CLNAlignment::detect(const cv::Mat& src,
                               const std::vector<cl::FaceBox>& windows,
                               std::vector<cl::FaceLandmark>& result) {
    cv::Mat img_gray;
    if (src.channels() != 1)
        cv::cvtColor(src, img_gray, cv::COLOR_BGR2GRAY);
    else
        img_gray = src;

    result.resize(windows.size());
    DetectLandmarksInVideo(img_gray, windows[0].bbox_, *clnf_model_.get(), model_param_);

    bool success = clnf_model_->detection_success;

    if(success) {
        int n = clnf_model_->patch_experts.visibilities[0][0].rows;
        result[0].points_.resize(n);
        for(int i = 0; i < n; ++i)
        {
            cv::Point2d landmark(clnf_model_->detected_landmarks.at<double>(i), clnf_model_->detected_landmarks.at<double>(i + n));
            result[0].points_[i] = landmark;
        }
    }
    return success;
}


// Pick only the more stable/rigid points under changes of expression
void CLNAlignment::extract_rigid_points(cv::Mat_<double>& source_points, cv::Mat_<double>& destination_points)
{
    if(source_points.rows == 68)
    {
        cv::Mat_<double> tmp_source = source_points.clone();
        source_points = cv::Mat_<double>();

        // Push back the rigid points (some face outline, eyes, and nose)
        source_points.push_back(tmp_source.row(1));
        source_points.push_back(tmp_source.row(2));
        source_points.push_back(tmp_source.row(3));
        source_points.push_back(tmp_source.row(4));
        source_points.push_back(tmp_source.row(12));
        source_points.push_back(tmp_source.row(13));
        source_points.push_back(tmp_source.row(14));
        source_points.push_back(tmp_source.row(15));
        source_points.push_back(tmp_source.row(27));
        source_points.push_back(tmp_source.row(28));
        source_points.push_back(tmp_source.row(29));
        source_points.push_back(tmp_source.row(31));
        source_points.push_back(tmp_source.row(32));
        source_points.push_back(tmp_source.row(33));
        source_points.push_back(tmp_source.row(34));
        source_points.push_back(tmp_source.row(35));
        source_points.push_back(tmp_source.row(36));
        source_points.push_back(tmp_source.row(39));
        source_points.push_back(tmp_source.row(40));
        source_points.push_back(tmp_source.row(41));
        source_points.push_back(tmp_source.row(42));
        source_points.push_back(tmp_source.row(45));
        source_points.push_back(tmp_source.row(46));
        source_points.push_back(tmp_source.row(47));

        cv::Mat_<double> tmp_dest = destination_points.clone();
        destination_points = cv::Mat_<double>();

        // Push back the rigid points
        destination_points.push_back(tmp_dest.row(1));
        destination_points.push_back(tmp_dest.row(2));
        destination_points.push_back(tmp_dest.row(3));
        destination_points.push_back(tmp_dest.row(4));
        destination_points.push_back(tmp_dest.row(12));
        destination_points.push_back(tmp_dest.row(13));
        destination_points.push_back(tmp_dest.row(14));
        destination_points.push_back(tmp_dest.row(15));
        destination_points.push_back(tmp_dest.row(27));
        destination_points.push_back(tmp_dest.row(28));
        destination_points.push_back(tmp_dest.row(29));
        destination_points.push_back(tmp_dest.row(31));
        destination_points.push_back(tmp_dest.row(32));
        destination_points.push_back(tmp_dest.row(33));
        destination_points.push_back(tmp_dest.row(34));
        destination_points.push_back(tmp_dest.row(35));
        destination_points.push_back(tmp_dest.row(36));
        destination_points.push_back(tmp_dest.row(39));
        destination_points.push_back(tmp_dest.row(40));
        destination_points.push_back(tmp_dest.row(41));
        destination_points.push_back(tmp_dest.row(42));
        destination_points.push_back(tmp_dest.row(45));
        destination_points.push_back(tmp_dest.row(46));
        destination_points.push_back(tmp_dest.row(47));
    }
}

bool CLNAlignment::is_valid_face(const cv::Vec6d& pose) const
{
    return (abs(pose[3]) < EulerThresh  && abs(pose[4] < EulerThresh) && abs(pose[5]) < EulerThresh);
}

bool CLNAlignment::load_model(const std::vector<std::string>& init_nets,
                                   const std::vector<std::string>& predict_nets)
{
    assert(predict_nets.size() == 1);
    clnf_model_ = make_shared<CLNF>(predict_nets[0]);
    return true;
}


