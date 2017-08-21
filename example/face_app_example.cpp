//
// Created by root on 17-8-18.
//

#include "cl_common.h"
#include "seeta_detector.h"
#include "openface_alignment.h"

using namespace cv;
using namespace std;
using namespace cl::fd;
using namespace cl::fa;
using namespace seeta::fd;
using namespace openface::fa;



int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0]
             << " detection_model alignment_model"
             << endl;
        return -1;
    }

    shared_ptr<Detector> face_detector = make_shared<SeetaDetector>(40, 40, 4, 4, 2.f, .5f);
    if(!face_detector->load_model(argv[1]))
    {
        cout <<"Can't load face detection model!" << endl;
        return  -1;
    }

    shared_ptr<Alignment> face_alignment = make_shared<OpenfaceAlignment>();
    if(!face_alignment->load_model(argv[2]))
    {
        cout <<"Can't load face alignment model!" << endl;
        return -2;
    }
    face_alignment->set_detector(face_detector.get());


    VideoCapture cap(0);
    if(!cap.isOpened())
    {
        cout <<"Can't open video!" << endl;
        return  -3;
    }

    for(;;)
    {
        Mat img;
        cap.read(img);
        if (img.empty()) break;


        Mat face_for_recognition, face_for_registor;
        cl::FaceInfo face;
        if(!face_alignment->detect(img, face, false)) {
            continue;
        } else{

//for registor only
            Rect large_bbx;
            int border_y = 80;
            int border_x = std::max((face.bbox_.height + 1.5f*border_y - face.bbox_.width)/2, 0.f);
            large_bbx.x = std::max(face.bbox_.x - border_x, 0);
            large_bbx.y = std::max(face.bbox_.y - border_y, 0);
            large_bbx.width = std::min<int>(face.bbox_.width + 2.0f*border_x, img.cols - large_bbx.x);
            large_bbx.height = std::min<int>(face.bbox_.height + 1.5f*border_y, img.rows - large_bbx.y);
            resize(img(large_bbx), face_for_registor, Size(cl::TrainWidth, cl::TrainHeight));

//for recognition only
            face_alignment->align(img, face, face_for_recognition);
        }


        imshow("src", img);
        imshow("face_reg", face_for_registor);
        imshow("face_rec", face_for_recognition);
        waitKey(2);
    }



}
