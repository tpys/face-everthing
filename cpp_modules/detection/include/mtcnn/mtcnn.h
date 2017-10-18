#ifndef CL_FACE_DETECTION_MTCNN_H
#define CL_FACE_DETECTION_MTCNN_H

#include "cl_common.h"
#include "cl_detector.h"
#include "cl_alignment.h"

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>


namespace mtcnn {
    namespace fd{


        class FaceDetector: public cl::fd::Detector, public cl::fa::Alignment{
        public:
            FaceDetector(int min_size=15,
                         float p_thresh=0.7,
                         float r_thresh=0.6,
                         float o_thresh=0.6,
                         float factor=0.709):min_size_(min_size),
                                             p_thresh_(p_thresh),
                                             r_thresh_(r_thresh),
                                             o_thresh_(o_thresh),
                                             factor_(factor) {
            }

            typedef struct _BBox {
                float xmin;
                float ymin;
                float xmax;
                float ymax;
                float score;
            } BBox;

            typedef struct _FaceInfo {
                float bbox_reg[4];
                float landmark_reg[10];
                float landmark[10];
                BBox bbox;
            } FaceInfo;


            bool detect(const cv::Mat& src, std::vector<cl::FaceBox>& faces);
            bool detect(const cv::Mat& src,
                        const std::vector<cl::FaceBox>& windows,
                        std::vector<cl::FaceLandmark>& landmarks);
            bool load_model(const std::vector<std::string>& init_nets,
                            const std::vector<std::string>& predict_nets);
            cv::Mat align_face(const cv::Mat& src,
                               const cl::FaceLandmark& landmark,
                               int width,
                               int height);


        private:
            std::vector<FaceInfo> Detect(const cv::Mat& img,
                                    const int min_size,
                                    const float* threshold,
                                    const float factor,
                                    const int stage);
            std::vector<FaceInfo> ProposalNet(const cv::Mat& img, int min_size, float threshold, float factor);
            std::vector<FaceInfo> NextStage(const cv::Mat& image,
                                       std::vector<FaceInfo> &pre_stage_res,
                                       int input_w,
                                       int input_h,
                                       int stage_num,
                                       const float threshold);
            void BBoxRegression(std::vector<FaceInfo>& bboxes);
            void BBoxPadSquare(std::vector<FaceInfo>& bboxes, int width, int height);
            void BBoxPad(std::vector<FaceInfo>& bboxes, int width, int height);
            void GenerateBBox(caffe::Blob<float>* confidence, caffe::Blob<float>* reg_box, float scale, float thresh);
            std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
            float IoU(float xmin, float ymin,
                      float xmax, float ymax,
                      float xmin_, float ymin_,
                      float xmax_, float ymax_,
                      bool is_iom = false);

            static bool CompareBBox(const FaceInfo & a, const FaceInfo & b) {
                return a.bbox.score > b.bbox.score;
            }

        public:
            const float PnetStride = 2;
            const float PnetCellSize = 12;
            const int PnetMaxDetectNum = 5000;
            const float MeanVal = 127.5f;
            const float StdVal = 0.0078125f;
            const int StepSize = 128;

            int min_size_;
            float p_thresh_;
            float r_thresh_;
            float o_thresh_;
            float factor_;

        private:
            boost::shared_ptr<caffe::Net<float>> PNet_;
            boost::shared_ptr<caffe::Net<float>> RNet_;
            boost::shared_ptr<caffe::Net<float>> ONet_;
            std::vector<FaceInfo> candidate_boxes_;
            std::vector<FaceInfo> total_boxes_;

        };




    }
}


#endif