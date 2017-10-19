//
// Created by root on 17-8-30.
//
#include "cl_common.h"
#include "mtcnn.h"

#include <fstream>

using namespace cv;
using namespace std;
using namespace cl::fd;
using namespace cl::fa;
namespace fs = boost::filesystem;


static std::vector<std::string> get_file_list(const std::string& path)
{
    vector<string> m_file_list;
    if (!path.empty())
    {
        fs::path apk_path(path);
        fs::recursive_directory_iterator end;

        for (fs::recursive_directory_iterator i(apk_path); i != end; ++i)
        {
            if (fs::is_regular_file(i->path())){
                const fs::path cp = (*i);
                m_file_list.push_back(cp.string());
            }
        }
    }
    return m_file_list;
}

Mat generate_thumbnail(const vector<string>& file_list, int rows, int cols, int size = 96){
    int length =  file_list.size();
    if(length < rows * cols){
        rows = sqrt(length);
        cols = sqrt(length);
    }
    Mat thumbnail_img(rows * size, cols * size, CV_8UC3, Scalar::all(255));
    int counter = 0;
    for(int r = 0; r < rows; ++ r){
        for(int c = 0; c < cols; ++ c){
            Mat src = imread(file_list[counter]);
            Mat thumbnail;
            resize(src, thumbnail, Size(size, size));
            thumbnail.copyTo(thumbnail_img(Rect(c*size, r*size, size, size)));
            ++counter;
            if(counter >= length) {
                break;
            }
        }
    }
    return thumbnail_img;
}


void test_humbnail(){
    vector<string> image_list = {
            "/home/tpys/dataset/0000045/001.jpg",
            "/home/tpys/dataset/0000045/002.jpg",
            "/home/tpys/dataset/0000045/003.jpg",
            "/home/tpys/dataset/0000045/004.jpg",
            "/home/tpys/dataset/0000045/005.jpg",
            "/home/tpys/dataset/0000045/006.jpg",
            "/home/tpys/dataset/0000045/007.jpg",
            "/home/tpys/dataset/0000045/008.jpg",
            "/home/tpys/dataset/0000045/009.jpg",
            "/home/tpys/dataset/0000045/011.jpg",
            "/home/tpys/dataset/0000045/012.jpg",
            "/home/tpys/dataset/0000045/013.jpg",
            "/home/tpys/dataset/0000045/014.jpg",
            "/home/tpys/dataset/0000045/015.jpg",
    };
    imwrite("aligned.jpg", generate_thumbnail(image_list, 3, 4, 96));
}

int main(int argc, char** argv) {

//    test_humbnail();return 0;

    /*must change these there lines to yours**/
    const string model_dir = "/home/tpys/projects/cl-face/trained_models/detection"; // mtcnn model dir
    const string dataset_dir = "/home/tpys/dataset/CASIA-maxpy-clean"; //dataset need to be align
    const string output_dir = "/home/tpys/dataset/CASIA-maxpy-clean-128";//output aligned dataset

    const int AlignWidth = 128;
    const int AlignHeight = 128;

    bool verbose = false;//turn on to visulize align result
    const vector<string> argvs = {
            model_dir + "/det1.prototxt",
            model_dir + "/det2.prototxt",
            model_dir + "/det3.prototxt",
            model_dir + "/det1.caffemodel",
            model_dir + "/det2.caffemodel",
            model_dir + "/det3.caffemodel",
    };

    std::shared_ptr<Alignment> face_alignment = std::make_shared<mtcnn::fd::FaceDetector>();
    if(!face_alignment->load_model({argvs[0], argvs[1], argvs[2]}, {argvs[3], argvs[4], argvs[5]}))
    {
        cout <<"Can't load face detection model!" << endl;
        return  -1;
    }

    ofstream file(output_dir + "/failed_list.txt");
    vector<string> failed_list;
    TickMeter tm;
    tm.start();
    auto image_list = get_file_list(dataset_dir);
    int64 total_num = image_list.size();
    fs::path dst_p(output_dir);
    int64 counter = 0;
    tm.stop();
    LOG(INFO) << "read list, time: " << tm.getTimeSec() << " seconds";



    tm.reset();tm.start();
    for(size_t i = 0; i < image_list.size(); ++i){
        fs::path src_p(image_list[i]);
        auto dir_path = dst_p / src_p.parent_path().filename();
        if(!fs::exists(dir_path)){
            fs::create_directories(dir_path);
        }

        Mat src = imread(image_list[i]);
        if (src.empty()) continue;

        Mat src_gray;
        Mat aligned_face;
        Mat src_display = src.clone();
        if(src.channels() == 3)
        {
            cv::cvtColor(src, src_gray, CV_BGR2GRAY);
        }

        vector<cl::FaceBox> face_boxes;
        vector<cl::FaceLandmark> face_landmarks;
        face_alignment->detect(src, face_boxes, face_landmarks);

        if(face_landmarks.size() > 0){
            auto biggest_id = cl::fa::get_biggest_id(face_landmarks, src.cols, src.rows);
            auto aligned_face = face_alignment->align_face(src, face_landmarks[biggest_id], AlignWidth, AlignHeight);

            auto full_name = dir_path/src_p.filename();
            imwrite(full_name.string(), aligned_face);
            ++counter;

            char msg[256];
            if(counter % 1000 == 0){
                sprintf(msg, "process: %d/%d", counter, total_num);
                LOG(INFO) << msg;
            }

            if(verbose){
                cl::fa::draw(face_boxes[biggest_id], face_landmarks[biggest_id], src_display);
                imshow("aligned", aligned_face);
                imshow("src", src_display);
                waitKey(0);
            }
        } else {
            LOG(INFO) << "Failed on: " << image_list[i];
            file << image_list[i] << endl;
            failed_list.push_back(image_list[i]);
        }
    }

    tm.stop();
    file.close();

    //the last one
    char msg[256];
    sprintf(msg, "process: %d/%d", counter, total_num);
    LOG(INFO) << msg;
    LOG(INFO) << "Align all dataset, time: " << tm.getTimeSec() << " seconds";

    LOG(INFO) << "Save thumbnail image";
    imwrite(output_dir + "/thumbnail.jpg", generate_thumbnail(failed_list, 40, 50, 24));

}



//452632/455594 half mtcnn onet for caisa-web-clean-data
//453078/455594 full mtcnn onet for caisa-web-clean-data