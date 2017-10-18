//
// Created by root on 17-9-26.
//
#include "tracker.hpp"
#include "openpose/face/faceParameters.hpp"

using namespace cl::ft;
using namespace cv;
using namespace std;

Tracker::Tracker(const string& pdm_model,
                 const float render_threshold,
                 cl::fa::Alignment* alignment) {
    pdm_.Read(pdm_model);
    face_alignment_.reset(alignment);
    render_threshold_ = render_threshold;
    params_local_.create(pdm_.NumberOfModes(), 1);
    reset();
}


bool Tracker::check_landmarks(const cl::FaceLandmark& landmarks){

    return true;
}


void Tracker::update_template(const cv::Mat& bgr_image){
    auto bbox = compute_bbox() & cv::Rect(0, 0, bgr_image.cols, bgr_image.rows);
    face_template_ = bgr_image(bbox).clone();
}


void Tracker::match_template(const cv::Mat& bgr_image){

    double scaling = param_.face_template_scale_;
    cv::Mat image;
//    if(scaling < 1)
//    {
//        cv::resize(face_template_, face_template_, cv::Size(), scaling, scaling);
//        cv::resize(bgr_image, image, cv::Size(), scaling, scaling);
//    }
//    else
    {
        scaling = 1;
        image = bgr_image.clone();
    }

    // Resizing the template
    cv::Mat corr_out;
    cv::Mat image_gray, face_template_gray;

    cv::cvtColor(image, image_gray, CV_BGR2BGRA);
    cv::cvtColor(face_template_, face_template_gray, CV_BGR2BGRA);
    cv::matchTemplate(image_gray, face_template_gray, corr_out, CV_TM_CCOEFF_NORMED);

    // Actually matching it
    //double min, max;
    int max_loc[2];
    cv::minMaxIdx(corr_out, NULL, NULL, NULL, max_loc);

    cv::Rect out_bbox(max_loc[1]/scaling,
                      max_loc[0]/scaling,
                      face_template_.rows / scaling,
                      face_template_.cols / scaling);

    face_rect_ = out_bbox & cv::Rect(0, 0, bgr_image.cols, bgr_image.rows);
}


void Tracker::reset(){
    param_.use_face_template_ = true;
    param_.face_template_scale_ = 1.f;
    tracking_success_ = false;
    initialised_ = false;
    params_local_.setTo(0.0);
    params_global_ = cv::Vec6d(1, 0, 0, 0, 0, 0);
    failures_ = -1;
    face_template_ = cv::Mat();
    face_rect_ = cv::Rect();
}


bool Tracker::tracking(const cv::Mat& bgr_image, const cv::Rect& bouding_box) {

    if(!initialised_ && bouding_box.width > 0){
        face_rect_ = bouding_box & cv::Rect(0, 0, bgr_image.cols, bgr_image.rows) ;
        initialised_ = true;
    }

    if(initialised_) {
        if(param_.use_face_template_ && !face_template_.empty() && tracking_success_){
            match_template(bgr_image);
        }

        cl::FaceBox window(face_rect_);
        result_.resize(1);
        tracking_success_ = face_alignment_->detect(bgr_image, { window }, result_);

        if(!check_landmarks(result_[0])){
            failures_++;
        } else{
            failures_--;
            update_template(bgr_image);
        }
        return tracking_success_;

    } else {
        return false;
    }
}



void Tracker::draw(cv::Mat& display_img) {

    int thickness = (int)std::ceil(3.0* ((double)display_img.cols) / 640.0);
    int thickness_2 = (int)std::ceil(1.0* ((double)display_img.cols) / 640.0);
    if(!face_rect_.empty())
        cv::rectangle(display_img, face_rect_, cv::Scalar(0, 255, 0), thickness);

    if(!result_.empty())
    for(auto& e: result_[0].points_){
        cv::circle(display_img, Point(e.x, e.y), 1 * DrawMultiplier, cv::Scalar(0, 0, 255), thickness, CV_AA, DrawShiftbits);
        cv::circle(display_img, Point(e.x, e.y), 1 * DrawMultiplier, cv::Scalar(255, 0, 0), thickness_2, CV_AA, DrawShiftbits);
    }
}