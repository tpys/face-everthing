#ifndef CL_FACE_DETECTION_MTCNN_H
#define CL_FACE_DETECTION_MTCNN_H

#include "cl_common.h"
#include "cl_detector.h"
#include "cl_alignment.h"

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
#include <type_traits>

//#define CPU_ONLY

namespace mtcnn {
    namespace fd{


            class FaceDetector: public cl::fd::Detector, public cl::fa::Alignment{
        public:
            struct FaceInfo{
            };
            enum COLOR_ORDER{
                GRAY,
                RGBA,
                RGB,
                BGRA,
                BGR
            };
            enum MODEL_VERSION{
                MODEL_V1,
                MODEL_V2
            };
            enum NMS_TYPE{
                MIN,
                UNION,
            };
            enum IMAGE_DIRECTION{
                ORIENT_LEFT,
                ORIENT_RIGHT,
                ORIENT_UP,
                ORIENT_DOWN,
            };
            struct BoundingBox{
                //rect two points
                float x1, y1;
                float x2, y2;
                //regression
                float dx1, dy1;
                float dx2, dy2;
                //cls
                float score;
                //inner points
                float points_x[5];
                float points_y[5];
            };

            struct CmpBoundingBox{
                bool operator() (const BoundingBox& b1, const BoundingBox& b2)
                {
                    return b1.score > b2.score;
                }
            };

            const float ScoreThresh = 0.8f;
        private:
            boost::shared_ptr<caffe::Net<float>> P_Net;
            boost::shared_ptr<caffe::Net<float>> R_Net;
            boost::shared_ptr<caffe::Net<float>> O_Net;
            //used by model 2 version
            boost::shared_ptr<caffe::Net<float>> L_Net;

            double                           img_mean;
            double                           img_var;
            cv::Size                         input_geometry_;
            int                              num_channels_;
            MODEL_VERSION                    model_version;
            int min_size_;
            float p_thresh_;
            float r_thresh_;
            float o_thresh_;
            float scale_;
        public:
            FaceDetector(int min_size=20,
                         float p_thresh=0.6,
                         float r_thresh=0.7,
                         float o_thresh=0.7,
                         float scale=0.709):min_size_(min_size),
                                            p_thresh_(p_thresh),
                                            r_thresh_(r_thresh),
                                            o_thresh_(o_thresh),
                                            scale_(scale) {

            }

            FaceDetector(const std::string& model_dir,
                         const MODEL_VERSION model_version);

            std::vector< BoundingBox > Detect (const cv::Mat& img,
                                          const COLOR_ORDER color_order,
                                          const IMAGE_DIRECTION orient,
                                          int min_size = 20,
                                          float P_thres = 0.6,
                                          float R_thres = 0.7,
                                          float O_thres =0.7,
                                          bool is_fast_resize = true,
                                          float scale_factor = 0.709);

            bool detect(const cv::Mat& src, std::vector<cl::FaceBox>& faces){

                auto result = Detect(src,
                                     FaceDetector::BGR,
                                     FaceDetector::ORIENT_UP,
                                     min_size_, p_thresh_, r_thresh_, o_thresh_, true, scale_);

                for(auto& e: result){
                    if (e.score > ScoreThresh){
                        faces.push_back(cl::FaceBox(e.x1, e.y1, e.x2, e.y2, e.score));
                    }
                }

                return true;
            }

            bool detect(const cv::Mat& src,
                        const std::vector<cl::FaceBox>& windows,
                        std::vector<cl::FaceLandmark>& landmarks) {

                auto result = Detect(src,
                                     FaceDetector::BGR,
                                     FaceDetector::ORIENT_UP,
                                     min_size_, p_thresh_, r_thresh_, o_thresh_, true, scale_);

                std::vector<cl::FaceBox>* face_boxes = const_cast<std::vector<cl::FaceBox>*>(&windows);

                for(auto& e: result){
                    if (e.score > ScoreThresh){
                        face_boxes->push_back(cl::FaceBox(e.x1, e.y1, e.x2, e.y2, e.score));
                        std::vector<cv::Point2f> keypoints = {
                                { e.points_x[0], e.points_y[0] },
                                { e.points_x[1], e.points_y[1] },
                                { e.points_x[2], e.points_y[2] },
                                { e.points_x[3], e.points_y[3] },
                                { e.points_x[4], e.points_y[4] },
                        };
                        std::vector<float> scores(5, e.score);
                        landmarks.push_back(cl::FaceLandmark(keypoints, scores));
                    }
                }

                return true;
            }

            bool load_model(const std::vector<std::string>& init_nets,
                                    const std::vector<std::string>& predict_nets){

                this->model_version = MODEL_V1;

#ifdef CPU_ONLY
                Caffe::set_mode(Caffe::CPU);
#else
                caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
                /* load three networks */
                //p
                P_Net.reset( new caffe::Net<float> (init_nets[0], caffe::TEST) );
                P_Net->CopyTrainedLayersFrom(predict_nets[0]);
                //R
                R_Net.reset( new caffe::Net<float> (init_nets[1], caffe::TEST) );
                R_Net->CopyTrainedLayersFrom(predict_nets[1]);
                //O
                O_Net.reset( new caffe::Net<float> (init_nets[2], caffe::TEST) );
                O_Net->CopyTrainedLayersFrom(predict_nets[2]);

                caffe::Blob<float>* input_layer = P_Net->input_blobs()[0];
                num_channels_ = input_layer->channels();
                input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
                //set img_mean
                img_mean = 127.5;
                //set img_var
                img_var  = 0.0078125;

                return true;
            }


            cv::Mat align_face(const cv::Mat& src,
                               const cl::FaceLandmark& landmark,
                               int width,
                               int height);



            cv::Size GetInputSize()   { return input_geometry_; }
            int      GetInputChannel(){ return num_channels_; }
            std::vector<int> GetInputShape()  {
                caffe::Blob<float>* input_layer = P_Net->input_blobs()[0];
                return input_layer->shape();
            }
        private:
            void generateBoundingBox(const std::vector<float>& boxRegs, const std::vector<int>& box_shape,
                                     const std::vector<float>& cls, const std::vector<int>& cls_shape,
                                     float scale, float threshold, std::vector<BoundingBox>& filterOutBoxes
            );
            void filteroutBoundingBox(const std::vector<BoundingBox>& boxes,
                                      const std::vector<float>& boxRegs, const std::vector<int>& box_shape,
                                      const std::vector<float>& cls, const std::vector<int>& cls_shape,
                                      const std::vector< float >& points, const std::vector< int >& points_shape,
                                      float threshold, std::vector<BoundingBox>& filterOutBoxes);
            void nms_cpu(std::vector<BoundingBox>& boxes, float threshold, NMS_TYPE type, std::vector<BoundingBox>& filterOutBoxes);

            //void pad(std::vector<BoundingBox>& boxes, int imgW, int imgH);

            //vector<int> nms(vector<int> boxs, );
            std::vector<float> predict(const cv::Mat& img);
            void wrapInputLayer(boost::shared_ptr<caffe::Net<float>> net, std::vector<cv::Mat>* input_channels);
            void pyrDown(const std::vector<cv::Mat>& img_channels,float scale, std::vector<cv::Mat>* input_channels);
            void buildInputChannels(const std::vector<cv::Mat>& img_channels, const std::vector<BoundingBox>& boxes,
                                    const cv::Size& target_size, std::vector<cv::Mat>* input_channels);

        };

    }
}


#endif