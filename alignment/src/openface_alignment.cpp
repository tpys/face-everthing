//
// Created by root on 17-8-18.
//

#include "openface_alignment.h"
#include "LandmarkDetectorFunc.h"

using namespace std;
using namespace openface::fa;
using namespace LandmarkDetector;

OpenfaceAlignment::OpenfaceAlignment(
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


bool OpenfaceAlignment::detect(cv::Mat& src, cl::FaceInfo& face, bool draw) {
    cv::Mat img_gray;
    if (src.channels() != 1)
        cv::cvtColor(src, img_gray, cv::COLOR_BGR2GRAY);
    else
        img_gray = src;

    cv::Rect_<double> bounding_box;
    DetectLandmarksInVideo(img_gray, bounding_box, *clnf_model_.get(), model_param_);
    bool success =  (clnf_model_->detection_success && (face.landmark_score_ < model_param_.validation_boundary));

    if(success)
    {
        int n = clnf_model_->patch_experts.visibilities[0][0].rows;
        face.landmarks_.resize(n);
        for(int i = 0; i < n; ++i)
        {
            cv::Point2d landmark(clnf_model_->detected_landmarks.at<double>(i), clnf_model_->detected_landmarks.at<double>(i + n));
            face.landmarks_[i] = landmark;
        }
        face.landmark_score_ = clnf_model_->detection_certainty;
        face.bbox_ = cv::boundingRect(face.landmarks_);

        if (cx_ == 0 || cy_ == 0)
        {
            cx_ = src.cols / 2.0f;
            cy_ = src.rows / 2.0f;
        }
        // Use a rough guess-timate of focal length
        if (fx_ == 0 || fy_ == 0)
        {
            fx_ = 500 * (src.cols / 640.0);
            fy_ = 500 * (src.rows / 480.0);

            fx_ = (fx_ + fy_) / 2.0;
            fy_ = fx_;
        }

        face.head_pose_ = GetCorrectedPoseWorld(*clnf_model_.get(), fx_, fy_, cx_, cy_);

//        cout << "pose: " << face.head_pose_ << endl;

        if(draw)
        {
            Draw(src, *clnf_model_.get());
        }

    }

    return (success && is_valid_face(face.head_pose_));
}

void OpenfaceAlignment::align(const cv::Mat& src, cl::FaceInfo& face, cv::Mat& dst)
{
    // Will warp to scaled mean shape
    cv::Mat_<double> similarity_normalised_shape = clnf_model_->pdm.mean_shape * SimScale;

    // Discard the z component
    similarity_normalised_shape = similarity_normalised_shape(cv::Rect(0, 0, 1, 2*similarity_normalised_shape.rows/3)).clone();

    cv::Mat_<double> source_landmarks = clnf_model_->detected_landmarks.reshape(1, 2).t();
    cv::Mat_<double> destination_landmarks = similarity_normalised_shape.reshape(1, 2).t();

    // Aligning only the more rigid points

    extract_rigid_points(source_landmarks, destination_landmarks);
    cv::Matx22d scale_rot_matrix = AlignShapesWithScale(source_landmarks, destination_landmarks);
    cv::Matx23d warp_matrix;

    warp_matrix(0,0) = scale_rot_matrix(0,0);
    warp_matrix(0,1) = scale_rot_matrix(0,1);
    warp_matrix(1,0) = scale_rot_matrix(1,0);
    warp_matrix(1,1) = scale_rot_matrix(1,1);

    double tx = clnf_model_->params_global[4];
    double ty = clnf_model_->params_global[5];

    cv::Vec2d T(tx, ty);
    T = scale_rot_matrix * T;

    // Make sure centering is correct
    warp_matrix(0,2) = -T(0) + cl::TestWidth/2;
    warp_matrix(1,2) = -T(1) + cl::TestHeight/2;

    cv::warpAffine(src, dst, warp_matrix, cv::Size(cl::TestWidth, cl::TestHeight), cv::INTER_LINEAR);

    // Move the destination landmarks there as well
    cv::Matx22d warp_matrix_2d(warp_matrix(0,0), warp_matrix(0,1), warp_matrix(1,0), warp_matrix(1,1));

    destination_landmarks = cv::Mat(clnf_model_->detected_landmarks.reshape(1, 2).t()) * cv::Mat(warp_matrix_2d).t();

    destination_landmarks.col(0) = destination_landmarks.col(0) + warp_matrix(0,2);
    destination_landmarks.col(1) = destination_landmarks.col(1) + warp_matrix(1,2);

    // Move the eyebrows up to include more of upper face
    destination_landmarks.at<double>(0,1) -= (30/0.7)*SimScale;
    destination_landmarks.at<double>(16,1) -= (30 / 0.7)*SimScale;
    destination_landmarks.at<double>(17,1) -= (30 / 0.7)*SimScale;
    destination_landmarks.at<double>(18,1) -= (30 / 0.7)*SimScale;
    destination_landmarks.at<double>(19,1) -= (30 / 0.7)*SimScale;
    destination_landmarks.at<double>(20,1) -= (30 / 0.7)*SimScale;
    destination_landmarks.at<double>(21,1) -= (30 / 0.7)*SimScale;
    destination_landmarks.at<double>(22,1) -= (30 / 0.7)*SimScale;
    destination_landmarks.at<double>(23,1) -= (30 / 0.7)*SimScale;
    destination_landmarks.at<double>(24,1) -= (30 / 0.7)*SimScale;
    destination_landmarks.at<double>(25,1) -= (30 / 0.7)*SimScale;
    destination_landmarks.at<double>(26,1) -= (30 / 0.7)*SimScale;
    destination_landmarks = cv::Mat(destination_landmarks.t()).reshape(1, 1).t();

//    face.patchs_[cl::FaceInfo::LEFT_EYE] = cv::Rect(0,0,0,0);

}


// Pick only the more stable/rigid points under changes of expression
void OpenfaceAlignment::extract_rigid_points(cv::Mat_<double>& source_points, cv::Mat_<double>& destination_points)
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

bool OpenfaceAlignment::is_valid_face(const cv::Vec6d& pose) const
{
    return (abs(pose[3]) < EulerThresh  && abs(pose[4] < EulerThresh) && abs(pose[5]) < EulerThresh);
}

bool OpenfaceAlignment::load_model(const std::string& model)
{
    clnf_model_ = make_shared<CLNF>(model);
    return true;
}


