//
// Created by root on 17-8-30.
//
#include "cl_common.h"
#include "cl_recognizer.h"
#include "sphereface.h"
#include "seeta_detector.h"
#include "LandmarkDetectorFunc.h"
#include "openpose/face/op_face.hpp"
#include "mtcnn.h"
#include "image_warping.h"

using namespace cv;
using namespace std;
using namespace cl::fd;
using namespace cl::fa;
using namespace cl::fr;
using namespace seeta::fd;
using namespace op::fa;
using namespace sf::fr;
using namespace cl::camera;
namespace fs = boost::filesystem;


#define FISHEYE_CAMERA

void remove_overlap_rect(const vector<shared_ptr< LandmarkDetector::CLNF>>& clnf_models,
                         vector<cl::FaceLandmark>& landmarks)
{
    const double thresh = 0.5f;
    for(size_t i = 0; i < clnf_models.size(); ++i)
    {
        cv::Rect_<double> model_rect = clnf_models[i]->GetBoundingBox();
        for(int j = landmarks.size()-1; j >=0; --j)
        {
            Rect_<double> landmark_rect = cl::fa::landmark2rect(landmarks[j]);
            double intersection_area = (model_rect & landmark_rect).area();
            double union_area = model_rect.area() + landmark_rect.area() - 2 * intersection_area;
            if( intersection_area/union_area > thresh)
            {
                landmarks.erase(landmarks.begin() + j);
            }
        }
    }
}


vector<cl::FaceBox> get_all_rect(const vector<shared_ptr< LandmarkDetector::CLNF>>& clnf_models){
    vector<cl::FaceBox> results(clnf_models.size());
    for(size_t i = 0; i < clnf_models.size(); ++i)
    {
        auto model_rect = clnf_models[i]->get_bbox();
        results[i] = (cl::FaceBox(model_rect, 1.f));
    }
    return results;
}


vector<pair<string, string>> load_facedb(const string& root_dir){
    vector<pair<string, string>> facedb;
    if (!root_dir.empty())
    {
        fs::path apk_path(root_dir);
        fs::recursive_directory_iterator end;

        for (fs::recursive_directory_iterator i(apk_path); i != end; ++i)
        {
            if (fs::is_regular_file(i->path())){
                const fs::path cp = (*i);
                facedb.push_back(make_pair(cp.parent_path().filename().string(), cp.string()));
            }
        }
    }
    return facedb;
};


struct TrackingInfo{

    bool tracking_ = false;
    bool recognized = false;
    int class_id_ = -1;
    string name_ = "unknown";

    void reset(){
        tracking_ = false;
        recognized = false;
        class_id_ = -1;
        name_ = "unknown";
    }
};



int main(int argc, char** argv) {

    //must change these lines to yours
    const string cln_model_dir =  "/home/tpys/projects/cl-face/trained_models/alignment/cln";
    const string mtcnn_model_dir = "/home/tpys/projects/cl-face/trained_models/detection";
    const string openpose_model_dir = "/home/tpys/projects/cl-face/trained_models/alignment/op_face";
    const string recognition_model_dir = "/home/tpys/projects/cl-face/trained_models/recognition";
    const string facedb_dir = "";
    const string video_dir = "";


    const vector<string> argvs = {
            cln_model_dir + "/main_clnf_general.txt",

            mtcnn_model_dir + "/det1.prototxt",
            mtcnn_model_dir + "/det2.prototxt",
            mtcnn_model_dir + "/det3.prototxt",
            mtcnn_model_dir + "/det1.caffemodel",
            mtcnn_model_dir + "/det2.caffemodel",
            mtcnn_model_dir + "/det3.caffemodel",

            openpose_model_dir + "/pose_deploy.prototxt",
            openpose_model_dir + "/pose_iter_116000.caffemodel",

            recognition_model_dir + "/sphereface_deploy.prototxt",
            recognition_model_dir + "/sphereface_model.caffemodel",
            recognition_model_dir + "/feature_mean.txt",
    };

    if (setenv ("DISPLAY", ":0", 0) == -1)
        return -1;

    /**must setting global variables*/
    const int MaxFaceNum = 4;
    const int DetectFrequency = 8;
    const int FaceWidth = 96;
    const int FaceHeight = 112;
    const double VisualisationBoundary = -0.8;
    bool verbose = false;
    bool enable_recognize = true;

    LandmarkDetector::FaceModelParameters model_param;
    model_param.reinit_video_every = 4;
    model_param.use_face_template = true;
    model_param.multi_view = false;
    model_param.sigma = 1.5;
    model_param.reg_factor = 25;
    model_param.weight_factor = 0;
    model_param.validation_boundary = -0.6;
    vector<TrackingInfo> tracking_info(MaxFaceNum);


    /**multi face tracking*/
    vector<shared_ptr< LandmarkDetector::CLNF>> clnf_models(MaxFaceNum);
    for(int i = 0; i < MaxFaceNum; ++i){
        clnf_models[i] = std::make_shared<LandmarkDetector::CLNF>(argvs[0]);
    }


    /**mtcnn is not only detector but also alignment*/
    std::shared_ptr<Alignment> face_alignment_1 = std::make_shared<mtcnn::fd::FaceDetector>();
    if(!face_alignment_1->load_model({argvs[1], argvs[2], argvs[3]}, {argvs[4], argvs[5], argvs[6]}))
    {
        cout <<"Can't load face detection model!" << endl;
        return  -1;
    }

    /**openpose face alignment*/
    std::shared_ptr<Alignment> face_alignment_2 = std::make_shared<OPFace>();
    if(!face_alignment_2->load_model({argvs[7]}, {argvs[8]}))
    {
        cout <<"Load alignment model failed!" << endl;
        return -2;
    }


    /**sphere face recognize*/
    std::shared_ptr<Recognizer> face_recognizer = std::make_shared<SphereFace>(1, 3, 112, 96, true);
    if(!face_recognizer->load_model({argvs[9]},{argvs[10]}))
    {
        cout <<"Can't load face detection model!" << endl;
        return  -3;
    }


    FaceClassifier face_classifier;
    if(!face_classifier.load_mean(argvs[11])){
        cout << "Can't load feature mean file" << endl;
        return -4;
    }

    auto facedb = load_facedb(facedb_dir);
    if(facedb.size() == 0) {
        enable_recognize = false;
    }
    for(size_t i = 0; i < facedb.size(); ++i) {
        auto src = imread(facedb[i].second);
        auto feature = face_recognizer->extract_feature(src);
        auto name = facedb[i].first;
        face_classifier.add_person(name, feature);
    }

    VideoCapture cap;
    if(!cap.open(video_dir)){
        cap.open(0);
    }
    if(!cap.isOpened())
    {
        cout <<"Can't open video!" << endl;
        return  -5;
    }
    namedWindow("tracking_result");
    moveWindow("tracking_result", 100, 100);


    int64 frame_count = 0;
    int64 t1,t0 = cv::getTickCount();
    double fps = 10;

#ifdef FISHEYE_CAMERA
    FisheyeWarping warping;
    Fisheye camera;
    Rect roi = Rect(0, 0, camera.output_width_, camera.output_height_);
    warping.init(&camera, roi);
#endif

    for(; ;)
    {
        Mat src;
        cap.read(src);

#ifdef FISHEYE_CAMERA
        warping.undistort_image(src, src);
        Rect detect_mask(0, src.rows/4, src.cols, src.rows/2);
#else
        Rect detect_mask(0, 0, src.cols, src.rows);
#endif

        TickMeter tm;
        cv::Mat src_display = src.clone();
        while(!src.empty()){
            src = src(detect_mask);

            cv::Mat_<uchar> src_gray;
            cv::Mat src_display = src.clone();

            if(src.channels() == 3)
            {
                cv::cvtColor(src, src_gray, CV_BGR2GRAY);
            }
            else
            {
                src_gray = src.clone();
            }

            vector<cl::FaceLandmark> five_init_landmarks;
            vector<cl::FaceLandmark> per_frame_landmarks;

            bool all_models_active = true;
            for(unsigned int i = 0; i < clnf_models.size(); ++i)
            {
                if(!tracking_info[i].tracking_)
                {
                    all_models_active = false;
                }
            }
            if(verbose){
                tm.reset();tm.start();
            }
            if(frame_count % DetectFrequency == 0 && !all_models_active)
            {
                if(face_alignment_1->detect(src, {}, five_init_landmarks)){
                    remove_overlap_rect(clnf_models, five_init_landmarks);
                }
            }

            vector<std::atomic<bool>> detections_used(five_init_landmarks.size());
            bool expected = true;

            if(verbose){
                tm.stop();
                LOG(INFO) << "Detection, time: " << tm.getTimeMilli() << " ms";
            }


            /**init tracker*/
            #pragma omp parallel for num_threads(CL_NUM_THREADS)
            for(unsigned int i = 0; i < clnf_models.size(); ++i) {
                if (clnf_models[i]->failures_in_a_row > 4) {
                    tracking_info[i].reset();
                    clnf_models[i]->Reset();
                }

                if (!tracking_info[i].tracking_) {
                    for (size_t j = 0; j < five_init_landmarks.size(); ++j) {
                          if(!detections_used[j].compare_exchange_weak(expected, false)) {
                            clnf_models[i]->Reset();
                            clnf_models[i]->detection_success = false;


                            init_tracker(src_gray,
                                         five_init_landmarks[j].points_,
                                         *clnf_models[i].get(),
                                         model_param);
                              tracking_info[i].tracking_ = true;
                              break;
                        }
                    }
                }
            }


            /**batch gpu alignment*/
            if(verbose){
                tm.reset();tm.start();
            }
            face_alignment_2->detect(src, get_all_rect(clnf_models), per_frame_landmarks);

            if(verbose){
                tm.stop();
                LOG(INFO) << "Alignment, time: " << tm.getTimeMilli() << " ms";
            }


            /**update tracker*/
            if(verbose){
                tm.reset();tm.start();
            }
            #pragma omp parallel for num_threads(CL_NUM_THREADS)
            for(unsigned int i = 0; i < clnf_models.size(); ++i) {
                if (tracking_info[i].tracking_) {
                    update_tracker(src_gray,
                                   per_frame_landmarks[i].points_,
                                   *clnf_models[i].get(),
                                   model_param);
                }
            }
            if(verbose){
                tm.stop();
                LOG(INFO) << "Tracking, time: " << tm.getTimeMilli() << " ms";
            }



            /**gpu recognize*/
            if(verbose){
                tm.reset();tm.start();
            }
            if(enable_recognize)
                for(unsigned int i = 0; i < clnf_models.size(); ++i) {
                    if (tracking_info[i].tracking_ &&  !tracking_info[i].recognized) {
                        Mat src_aligned = face_alignment_1->align_face(src,
                                                                     per_frame_landmarks[i],
                                                                     FaceWidth,
                                                                     FaceHeight);

                        auto feature = face_recognizer->extract_feature(src_aligned);
                        tracking_info[i].class_id_ = face_classifier.identify(feature);
                        tracking_info[i].name_ = face_classifier.get_name(tracking_info[i].class_id_);

                        if(tracking_info[i].class_id_ != -1) {
                            tracking_info[i].recognized = true;
                        }
                    }
                }
            if(verbose){
                tm.stop();
                LOG(INFO) << "Recognition, time: " << tm.getTimeMilli() << " ms";
            }


            // Work out the framerate
            if(frame_count % 10 == 0)
            {
                t1 = cv::getTickCount();
                fps = 10.0 / (double(t1-t0)/cv::getTickFrequency());
                t0 = t1;
            }
            char fpsC[255];
            sprintf(fpsC, "%d", (int)fps);
            string fpsSt("FPS:");
            fpsSt += fpsC;
            cv::putText(src_display, fpsSt, cv::Point(10,20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 1, CV_AA);


            // Go through every model and visualise the results
            for(int model = 0; model < clnf_models.size(); ++model)
            {
                if (tracking_info[model].tracking_){

                    if(clnf_models[model]->detection_certainty < VisualisationBoundary)
                    {
                        Mat src_display_temp = src_display(detect_mask);
                        LandmarkDetector::Draw(src_display_temp, *clnf_models[model].get());
                    }

                    string full_msg = "";
                    if(tracking_info[model].recognized){
                        full_msg += "Name: " + tracking_info[model].name_;
                    } else{
                        full_msg += "Name: unknown";
                    }
                    cv::Rect_<double> model_rect = clnf_models[model]->GetBoundingBox();
                    Point position(model_rect.x, std::max<int>(0, (model_rect.y-0.2*model_rect.height)));
                    float font_scale = 0.5;
                    cv::putText(src_display(detect_mask),
                                full_msg,
                                position,
                                CV_FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                CV_RGB(255, 255, 0),
                                2,
                                CV_AA);
                }
            }

            imshow("tracking_result", src_display);
            cap.read(src);

#ifdef FISHEYE_CAMERA
            warping.undistort_image(src, src);
#endif
            src_display.release();
            src_display = src.clone();

            char key = waitKey(2);
            if(key == 27){
                break;
            }
            frame_count++;
        }




        //ToDo clear memory
        frame_count = 0;
        for(size_t model=0; model < clnf_models.size(); ++model)
        {
            clnf_models[model]->Reset();
            tracking_info[model].reset();
        }
    }
}
