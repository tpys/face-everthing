//
// Created by root on 17-8-18.
//

#include "cl_common.h"
#include "seeta_detector.h"
#include "mtcnn.h"
#include "cln_alignment.h"
#include "es_alignment.h"
#include "lbf_alignment.h"
#include "openpose/face/op_face.hpp"

using namespace cv;
using namespace std;

using namespace cl::fd;
using namespace cl::fa;
using namespace seeta::fd;
using namespace mtcnn::fd;
using namespace cln::fa;
using namespace es::fa;
using namespace lbf::fa;
using namespace op::fa;

#define CLN 0
#define ES 0
#define LBF 0
#define OPENPOSE 1

int main(int argc, char** argv) {

    argv[1] = "/home/tpys/projects/cl-face/trained_models/detection/seeta_fd_frontal_v1.0.bin";
    argv[2] = "/home/tpys/projects/cl-face/trained_models/alignment/cln/main_clnf_general.txt";
    argv[3] = "/home/tpys/projects/cl-face/trained_models/alignment/es/model.txt";
    argv[4] = "/home/tpys/projects/cl-face/trained_models/alignment/lbf/LBF.model";
    argv[5] = "/home/tpys/projects/cl-face/trained_models/alignment/lbf/Regressor.model";
    argv[6] = "/home/tpys/projects/cl-face/trained_models/alignment/op_face/pose_deploy.prototxt";
    argv[7] = "/home/tpys/projects/cl-face/trained_models/alignment/op_face/pose_iter_116000.caffemodel";

    /**for mtcnn model*/
    argv[8] = "/home/tpys/projects/cl-face/trained_models/detection/det1.prototxt";
    argv[9] = "/home/tpys/projects/cl-face/trained_models/detection/det2.prototxt";
    argv[10] = "/home/tpys/projects/cl-face/trained_models/detection/det3.prototxt";
    argv[11] = "/home/tpys/projects/cl-face/trained_models/detection/det1.caffemodel";
    argv[12] = "/home/tpys/projects/cl-face/trained_models/detection/det2.caffemodel";
    argv[13] = "/home/tpys/projects/cl-face/trained_models/detection/det3.caffemodel";



    /**mtcnn face detector*/
    std::shared_ptr<Detector> face_detector = std::make_shared<mtcnn::fd::FaceDetector>();
    if(!face_detector->load_model({argv[8], argv[9], argv[10]},{argv[11], argv[12], argv[13]}))
    {
        cout <<"Can't load face detection model!" << endl;
        return  -1;
    }



#if CLN
    std::shared_ptr<Alignment> face_alignment = std::make_shared<CLNAlignment>();
    if(!face_alignment->load_model({}, {argv[2]}))
    {
        cout <<"Can't load face alignment model!" << endl;
        return -2;
    }
#elif ES
    std::shared_ptr<Alignment> face_alignment = std::make_shared<ExplicitShape>();
    if(!face_alignment->load_model({}, {argv[3]}))
    {
        cout <<"Can't load face alignment model!" << endl;
        return -2;
    }
#elif LBF
    std::shared_ptr<Alignment> face_alignment = std::make_shared<LBFAlignment>();
    if(!face_alignment->load_model({argv[4]}, {argv[5]}))
    {
        cout <<"Can't load face alignment model!" << endl;
        return -2;
    }
#elif OPENPOSE
    std::shared_ptr<Alignment> face_alignment = std::make_shared<OPFace>();
    if(!face_alignment->load_model({argv[6]}, {argv[7]}))
    {
        cout <<"Load alignment model failed!" << endl;
        return -2;
    }
#endif


    VideoCapture cap(0);
    if(!cap.isOpened())
    {
        cout <<"Can't open video!" << endl;
        return  -3;
    }
    namedWindow("detection");

    for(;;)
    {
        Mat img;
        cap.read(img);
        if (img.empty()) continue;

        Mat display_img = img.clone();
        vector<cl::FaceBox> windows;
        vector<cl::FaceLandmark> landmarks;

        /**detect */
        if(face_detector->detect(img, windows)) {
            for(const auto& e: windows){
                cv::rectangle(display_img, e.bbox_, cv::Scalar(0, 255, 0), 1);
            }
        }



        /**alignment*/
        if(face_alignment->detect(img, windows, landmarks)){
            for(const auto& face: landmarks){
                auto bounding_box = cv::boundingRect(face.points_);

                double scale = 0.2;
                double x = std::max<double>(0, bounding_box.x - scale*bounding_box.width/2);
                double y = std::max<double>(0, bounding_box.y - scale*bounding_box.width);
                double width = (1 + scale)*bounding_box.width;
                double height = (1 + scale)*bounding_box.width;

                bounding_box = {
                        x,
                        y,
                        width,
                        height,
                };

                cv::rectangle(display_img,bounding_box, cv::Scalar(255, 0, 0), 2);

                for(const auto& e: face.points_){
                    cv::circle(display_img, Point(e.x, e.y), 1, cv::Scalar(0, 0, 255), 2);
                }
            }
        }


        cv::imshow("detection", display_img);
        char key = waitKey(2);
        if(key == 27){
            break;
        }
    }
}

