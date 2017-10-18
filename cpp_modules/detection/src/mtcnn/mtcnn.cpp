//
// Created by root on 17-9-27.
//
#include "mtcnn.h"
using namespace cv;
using namespace std;
using namespace caffe;

using FaceInfo = mtcnn::fd::FaceDetector::FaceInfo;

namespace mtcnn{
    namespace fd{

        const std::vector<unsigned> ReferenceID = {
                36, 45, 30, 48, 54,
        };

        Mat tformfwd(const Mat& trans, const Mat& uv){
            Mat uv_h = Mat::ones(uv.rows, 3, CV_64FC1);
            uv.copyTo(uv_h(Rect(0, 0, 2, uv.rows)));
            Mat xv_h = uv_h*trans;
            return xv_h(Rect(0, 0, 2, uv.rows));
        }

        Mat find_none_flectives_similarity(const Mat& uv, const Mat& xy){
            Mat A = Mat::zeros(2*xy.rows, 4, CV_64FC1);
            Mat b = Mat::zeros(2*xy.rows, 1, CV_64FC1);
            Mat x = Mat::zeros(4, 1, CV_64FC1);

            xy(Rect(0, 0, 1, xy.rows)).copyTo(A(Rect(0, 0, 1, xy.rows)));//x
            xy(Rect(1, 0, 1, xy.rows)).copyTo(A(Rect(1, 0, 1, xy.rows)) );//y
            A(Rect(2, 0, 1, xy.rows)).setTo(1.);

            xy(Rect(1, 0, 1, xy.rows)).copyTo(A(Rect(0, xy.rows, 1, xy.rows)));//y
            (xy(Rect(0, 0, 1, xy.rows))).copyTo(A(Rect(1, xy.rows, 1, xy.rows)));//-x
            A(Rect(1, xy.rows, 1, xy.rows)) *= -1;
            A(Rect(3, xy.rows, 1, xy.rows)).setTo(1.);

            uv(Rect(0, 0, 1, uv.rows)).copyTo(b(Rect(0, 0, 1, uv.rows)));
            uv(Rect(1, 0, 1, uv.rows)).copyTo(b(Rect(0, uv.rows, 1, uv.rows)));

            cv::solve(A, b, x, cv::DECOMP_SVD);
            Mat trans_inv = (Mat_<double>(3, 3) << x.at<double>(0),  -x.at<double>(1), 0,
                    x.at<double>(1),  x.at<double>(0), 0,
                    x.at<double>(2),  x.at<double>(3), 1);
            Mat trans = trans_inv.inv(cv::DECOMP_SVD);
            trans.at<double>(0, 2) = 0;
            trans.at<double>(1, 2) = 0;
            trans.at<double>(2, 2) = 1;

            return trans;
        }

        Mat find_similarity(const Mat& uv, const Mat& xy){
            Mat trans1 =find_none_flectives_similarity(uv, xy);
            Mat xy_reflect = xy;
            xy_reflect(Rect(0, 0, 1, xy.rows)) *= -1;
            Mat trans2r = find_none_flectives_similarity(uv, xy_reflect);
            Mat reflect = (Mat_<double>(3, 3) << -1, 0, 0, 0, 1, 0, 0, 0, 1);

            Mat trans2 = trans2r*reflect;
            Mat xy1 = tformfwd(trans1, uv);

            double norm1 = cv::norm(xy1 - xy);

            Mat xy2 = tformfwd(trans2, uv);
            double norm2 = cv::norm(xy2 - xy);

            Mat trans;
            if(norm1 < norm2){
                trans = trans1;
            } else {
                trans = trans2;
            }
            return trans;
        }

        Mat get_similarity_transform(const vector<Point2f>& src_points, const vector<Point2f>& dst_points, bool reflective = true){
            Mat trans;
            Mat src((int)src_points.size(), 2, CV_32FC1, (void*)(&src_points[0].x));
            src.convertTo(src, CV_64FC1);

            Mat dst((int)dst_points.size(), 2, CV_32FC1, (void*)(&dst_points[0].x));
            dst.convertTo(dst, CV_64FC1);

            if(reflective){
                trans = find_similarity(src, dst);
            } else {
                trans = find_none_flectives_similarity(src, dst);
            }
            Mat trans_cv2 = trans(Rect(0, 0, 2, trans.rows)).t();

            return trans_cv2;
        }


        cv::Mat FaceDetector::align_face(const cv::Mat& src,
                                         const cl::FaceLandmark& landmark,
                                         int width,
                                         int height){

            const int N = landmark.points_.size();
            vector<Point2f> detect_points;
            if(N != 5){
                detect_points = {
                        landmark.points_[ReferenceID[0]],
                        landmark.points_[ReferenceID[1]],
                        landmark.points_[ReferenceID[2]],
                        landmark.points_[ReferenceID[3]],
                        landmark.points_[ReferenceID[4]],
                };
            } else{
                detect_points = landmark.points_;
            }

            const int ReferenceWidth = 96;
            const int ReferenceHeight = 112;
            vector<Point2f> reference_points = {
                    {30.29459953,  51.69630051},
                    {65.53179932,  51.50139999},
                    {48.02519989,  71.73660278},
                    {33.54930115,  92.3655014},
                    {62.72990036,  92.20410156}
            };
            for(auto& e: reference_points){
                e.x += (width - ReferenceWidth)/2.0f;
                e.y += (height - ReferenceHeight)/2.0f;
            }
            Mat tfm = get_similarity_transform(detect_points, reference_points);
            Mat aligned_face;
            warpAffine(src, aligned_face, tfm, Size(width, height));
            return aligned_face;
        }

        bool FaceDetector::detect(const cv::Mat& src, std::vector<cl::FaceBox>& faces){

            float threshold[] = { p_thresh_, r_thresh_, o_thresh_ };
            auto result = Detect(src,
                                 min_size_,
                                 threshold,
                                 factor_,
                                 3);

            for(auto& e: result){
                auto bbox = e.bbox;
                faces.push_back(cl::FaceBox(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.score));
            }
            return true;
        }



        bool FaceDetector::detect(const cv::Mat& src,
                                  const std::vector<cl::FaceBox>& windows,
                                  std::vector<cl::FaceLandmark>& landmarks) {

            float threshold[] = { p_thresh_, r_thresh_, o_thresh_ };
            auto result = Detect(src,
                                 min_size_,
                                 threshold,
                                 factor_,
                                 3);

            std::vector<cl::FaceBox>* face_boxes = const_cast<std::vector<cl::FaceBox>*>(&windows);
            for(auto& e: result){
                auto bbox = e.bbox;
                auto landmark = e.landmark;
                face_boxes->push_back(cl::FaceBox(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.score));
                std::vector<cv::Point2f> keypoints = {
                        { landmark[0], landmark[1] },
                        { landmark[2], landmark[3] },
                        { landmark[4], landmark[5] },
                        { landmark[6], landmark[7] },
                        { landmark[8], landmark[9] },
                };
                landmarks.push_back(cl::FaceLandmark(keypoints));
            }
            return true;
        }

        bool FaceDetector::load_model(const std::vector<std::string>& init_nets,
                                      const std::vector<std::string>& predict_nets){

            Caffe::set_mode(Caffe::GPU);
            PNet_.reset(new Net<float>(init_nets[0], TEST));
            PNet_->CopyTrainedLayersFrom(predict_nets[0]);
            RNet_.reset(new Net<float>(init_nets[1], TEST));
            RNet_->CopyTrainedLayersFrom(predict_nets[1]);
            ONet_.reset(new Net<float>(init_nets[2], TEST));
            ONet_->CopyTrainedLayersFrom(predict_nets[2]);

            Blob<float>* input_layer;
            input_layer = PNet_->input_blobs()[0];
            int num_channels_ = input_layer->channels();
            CHECK(num_channels_ == 3) << "Input layer should have 3 channels.";

            return true;
        }


        float FaceDetector::IoU(float xmin, float ymin, float xmax, float ymax,
                                float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom) {
            float iw = std::min(xmax, xmax_) - std::max(xmin, xmin_) + 1;
            float ih = std::min(ymax, ymax_) - std::max(ymin, ymin_) + 1;
            if (iw <= 0 || ih <= 0)
                return 0;
            float s = iw*ih;
            if (is_iom) {
                float ov = s / min((xmax - xmin + 1)*(ymax - ymin + 1), (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1));
                return ov;
            }
            else {
                float ov = s / ((xmax - xmin + 1)*(ymax - ymin + 1) + (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1) - s);
                return ov;
            }
        }
        std::vector<FaceInfo> FaceDetector::NMS(std::vector<FaceInfo>& bboxes,
                                                float thresh, char methodType) {
            std::vector<FaceInfo> bboxes_nms;
            if (bboxes.size() == 0) {
                return bboxes_nms;
            }
            std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

            int32_t select_idx = 0;
            int32_t num_bbox = static_cast<int32_t>(bboxes.size());
            std::vector<int32_t> mask_merged(num_bbox, 0);
            bool all_merged = false;

            while (!all_merged) {
                while (select_idx < num_bbox && mask_merged[select_idx] == 1)
                    select_idx++;
                if (select_idx == num_bbox) {
                    all_merged = true;
                    continue;
                }

                bboxes_nms.push_back(bboxes[select_idx]);
                mask_merged[select_idx] = 1;

                BBox select_bbox = bboxes[select_idx].bbox;
                float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin + 1) * (select_bbox.ymax - select_bbox.ymin + 1));
                float x1 = static_cast<float>(select_bbox.xmin);
                float y1 = static_cast<float>(select_bbox.ymin);
                float x2 = static_cast<float>(select_bbox.xmax);
                float y2 = static_cast<float>(select_bbox.ymax);

                select_idx++;
#pragma omp parallel for num_threads(CL_NUM_THREADS)
                for (int32_t i = select_idx; i < num_bbox; i++) {
                    if (mask_merged[i] == 1)
                        continue;

                    BBox & bbox_i = bboxes[i].bbox;
                    float x = std::max<float>(x1, static_cast<float>(bbox_i.xmin));
                    float y = std::max<float>(y1, static_cast<float>(bbox_i.ymin));
                    float w = std::min<float>(x2, static_cast<float>(bbox_i.xmax)) - x + 1;
                    float h = std::min<float>(y2, static_cast<float>(bbox_i.ymax)) - y + 1;
                    if (w <= 0 || h <= 0)
                        continue;

                    float area2 = static_cast<float>((bbox_i.xmax - bbox_i.xmin + 1) * (bbox_i.ymax - bbox_i.ymin + 1));
                    float area_intersect = w * h;

                    switch (methodType) {
                        case 'u':
                            if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
                                mask_merged[i] = 1;
                            break;
                        case 'm':
                            if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
                                mask_merged[i] = 1;
                            break;
                        default:
                            break;
                    }
                }
            }
            return bboxes_nms;
        }
        void FaceDetector::BBoxRegression(vector<FaceInfo>& bboxes) {
#pragma omp parallel for num_threads(CL_NUM_THREADS)
            for (int i = 0; i < bboxes.size(); ++i) {
                BBox &bbox = bboxes[i].bbox;
                float *bbox_reg = bboxes[i].bbox_reg;
                float w = bbox.xmax - bbox.xmin + 1;
                float h = bbox.ymax - bbox.ymin + 1;
                bbox.xmin += bbox_reg[0] * w;
                bbox.ymin += bbox_reg[1] * h;
                bbox.xmax += bbox_reg[2] * w;
                bbox.ymax += bbox_reg[3] * h;
            }
        }
        void FaceDetector::BBoxPad(vector<FaceInfo>& bboxes, int width, int height) {
#pragma omp parallel for num_threads(CL_NUM_THREADS)
            for (int i = 0; i < bboxes.size(); ++i) {
                BBox &bbox = bboxes[i].bbox;
                bbox.xmin = round(max(bbox.xmin, 0.f));
                bbox.ymin = round(max(bbox.ymin, 0.f));
                bbox.xmax = round(min(bbox.xmax, width - 1.f));
                bbox.ymax = round(min(bbox.ymax, height - 1.f));
            }
        }
        void FaceDetector::BBoxPadSquare(vector<FaceInfo>& bboxes, int width, int height) {
#pragma omp parallel for num_threads(CL_NUM_THREADS)
            for (int i = 0; i < bboxes.size(); ++i) {
                BBox &bbox = bboxes[i].bbox;
                float w = bbox.xmax - bbox.xmin + 1;
                float h = bbox.ymax - bbox.ymin + 1;
                float side = h>w ? h : w;
                bbox.xmin = round(max(bbox.xmin + (w - side)*0.5f, 0.f));

                bbox.ymin = round(max(bbox.ymin + (h - side)*0.5f, 0.f));
                bbox.xmax = round(min(bbox.xmin + side - 1, width - 1.f));
                bbox.ymax = round(min(bbox.ymin + side - 1, height - 1.f));
            }
        }
        void FaceDetector::GenerateBBox(Blob<float>* confidence, Blob<float>* reg_box,
                                        float scale, float thresh) {
            int feature_map_w_ = confidence->width();
            int feature_map_h_ = confidence->height();
            int spatical_size = feature_map_w_*feature_map_h_;
            const float* confidence_data = confidence->cpu_data() + spatical_size;
            const float* reg_data = reg_box->cpu_data();
            candidate_boxes_.clear();
            for (int i = 0; i<spatical_size; i++) {
                if (confidence_data[i] >= thresh) {

                    int y = i / feature_map_w_;
                    int x = i - feature_map_w_ * y;
                    FaceInfo faceInfo;
                    BBox &faceBox = faceInfo.bbox;

                    faceBox.xmin = (float)(x * PnetStride) / scale;
                    faceBox.ymin = (float)(y * PnetStride) / scale;
                    faceBox.xmax = (float)(x * PnetStride + PnetCellSize - 1.f) / scale;
                    faceBox.ymax = (float)(y * PnetStride + PnetCellSize - 1.f) / scale;

                    faceInfo.bbox_reg[0] = reg_data[i];
                    faceInfo.bbox_reg[1] = reg_data[i + spatical_size];
                    faceInfo.bbox_reg[2] = reg_data[i + 2 * spatical_size];
                    faceInfo.bbox_reg[3] = reg_data[i + 3 * spatical_size];

                    faceBox.score = confidence_data[i];
                    candidate_boxes_.push_back(faceInfo);
                }
            }
        }


        vector<FaceInfo> FaceDetector::ProposalNet(const cv::Mat& img, int minSize, float threshold, float factor) {
            cv::Mat  resized;
            int width = img.cols;
            int height = img.rows;
            float scale = 12.f / minSize;
            float minWH = std::min(height, width) *scale;
            std::vector<float> scales;
            while (minWH >= 12) {
                scales.push_back(scale);
                minWH *= factor;
                scale *= factor;
            }
            Blob<float>* input_layer = PNet_->input_blobs()[0];
            total_boxes_.clear();
            for (int i = 0; i < scales.size(); i++) {
                int ws = (int)std::ceil(width*scales[i]);
                int hs = (int)std::ceil(height*scales[i]);
                cv::resize(img, resized, cv::Size(ws, hs), 0, 0, cv::INTER_LINEAR);
                input_layer->Reshape(1, 3, hs, ws);
                PNet_->Reshape();
                float * input_data = input_layer->mutable_cpu_data();
                cv::Vec3b * img_data = (cv::Vec3b *)resized.data;
                int spatial_size = ws* hs;
                for (int k = 0; k < spatial_size; ++k) {
                    input_data[k] = float((img_data[k][0] - MeanVal)* StdVal);
                    input_data[k + spatial_size] = float((img_data[k][1] - MeanVal) * StdVal);
                    input_data[k + 2 * spatial_size] = float((img_data[k][2] - MeanVal) * StdVal);
                }
                PNet_->Forward();

                Blob<float>* confidence = PNet_->blob_by_name("prob1").get();
                Blob<float>* reg = PNet_->blob_by_name("conv4-2").get();
                GenerateBBox(confidence, reg, scales[i], threshold);
                std::vector<FaceInfo> bboxes_nms = NMS(candidate_boxes_, 0.5, 'u');
                if (bboxes_nms.size()>0) {
                    total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
                }
            }
            int num_box = (int)total_boxes_.size();
            vector<FaceInfo> res_boxes;
            if (num_box != 0) {
                res_boxes = NMS(total_boxes_, 0.7f, 'u');
                BBoxRegression(res_boxes);
                BBoxPadSquare(res_boxes, width, height);
            }
            return res_boxes;
        }
        vector<FaceInfo> FaceDetector::NextStage(const cv::Mat& image, vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold) {
            vector<FaceInfo> res;
            int batch_size = (int)pre_stage_res.size();
            if (batch_size == 0)
                return res;
            Blob<float>* input_layer = nullptr;
            Blob<float>* confidence = nullptr;
            Blob<float>* reg_box = nullptr;
            Blob<float>* reg_landmark = nullptr;

            switch (stage_num) {
                case 2: {
                    input_layer = RNet_->input_blobs()[0];
                    input_layer->Reshape(batch_size, 3, input_h, input_w);
                    RNet_->Reshape();
                }break;
                case 3: {
                    input_layer = ONet_->input_blobs()[0];
                    input_layer->Reshape(batch_size, 3, input_h, input_w);
                    ONet_->Reshape();
                }break;
                default:
                    return res;
                    break;
            }
            float * input_data = input_layer->mutable_cpu_data();
            int spatial_size = input_h*input_w;

#pragma omp parallel for num_threads(CL_NUM_THREADS)
            for (int n = 0; n < batch_size; ++n) {
                BBox &box = pre_stage_res[n].bbox;
                Mat roi = image(Rect(Point((int)box.xmin, (int)box.ymin), Point((int)box.xmax, (int)box.ymax))).clone();
                resize(roi, roi, Size(input_w, input_h));
                float *input_data_n = input_data + input_layer->offset(n);
                Vec3b *roi_data = (Vec3b *)roi.data;
                CHECK_EQ(roi.isContinuous(), true);
                for (int k = 0; k < spatial_size; ++k) {
                    input_data_n[k] = float((roi_data[k][0] - MeanVal)*StdVal);
                    input_data_n[k + spatial_size] = float((roi_data[k][1] - MeanVal)*StdVal);
                    input_data_n[k + 2 * spatial_size] = float((roi_data[k][2] - MeanVal)*StdVal);
                }
            }
            switch (stage_num) {
                case 2: {
                    RNet_->Forward();
                    confidence = RNet_->blob_by_name("prob1").get();
                    reg_box = RNet_->blob_by_name("conv5-2").get();
                }break;
                case 3: {
                    ONet_->Forward();
                    confidence = ONet_->blob_by_name("prob1").get();
                    reg_box = ONet_->blob_by_name("conv6-2").get();
                    reg_landmark = ONet_->blob_by_name("conv6-3").get();
                }break;
            }
            const float* confidence_data = confidence->cpu_data();
            const float* reg_data = reg_box->cpu_data();
            const float* landmark_data = nullptr;
            if (reg_landmark) {
                landmark_data = reg_landmark->cpu_data();
            }
            for (int k = 0; k < batch_size; ++k) {
                if (confidence_data[2 * k + 1] >= threshold) {
                    FaceInfo info;
                    info.bbox.score = confidence_data[2 * k + 1];
                    info.bbox.xmin = pre_stage_res[k].bbox.xmin;
                    info.bbox.ymin = pre_stage_res[k].bbox.ymin;
                    info.bbox.xmax = pre_stage_res[k].bbox.xmax;
                    info.bbox.ymax = pre_stage_res[k].bbox.ymax;
                    for (int i = 0; i < 4; ++i) {
                        info.bbox_reg[i] = reg_data[4 * k + i];
                    }
                    if (reg_landmark) {
                        float w = info.bbox.xmax - info.bbox.xmin + 1.f;
                        float h = info.bbox.ymax - info.bbox.ymin + 1.f;
                        for (int i = 0; i < 5; ++i){
                            info.landmark[2 * i] = landmark_data[10 * k + 2 * i] * w + info.bbox.xmin;
                            info.landmark[2 * i + 1] = landmark_data[10 * k + 2 * i + 1] * h + info.bbox.ymin;
                        }
                    }
                    res.push_back(info);
                }
            }
            return res;
        }
        vector<FaceInfo> FaceDetector::Detect(const cv::Mat& image, const int minSize, const float* threshold, const float factor, const int stage) {
            vector<FaceInfo> pnet_res;
            vector<FaceInfo> rnet_res;
            vector<FaceInfo> onet_res;
            if (stage >= 1){
                pnet_res = ProposalNet(image, minSize, threshold[0], factor);
            }
            if (stage >= 2 && pnet_res.size()>0){
                if (PnetMaxDetectNum < (int)pnet_res.size()){
                    pnet_res.resize(PnetMaxDetectNum);
                }
                int num = (int)pnet_res.size();
                int size = (int)ceil(1.f*num / StepSize);
                for (int iter = 0; iter < size; ++iter){
                    int start = iter*StepSize;
                    int end = min(start + StepSize, num);
                    vector<FaceInfo> input(pnet_res.begin() + start, pnet_res.begin() + end);
                    vector<FaceInfo> res = NextStage(image, input, 24, 24, 2, threshold[1]);
                    rnet_res.insert(rnet_res.end(), res.begin(), res.end());
                }
                rnet_res = NMS(rnet_res, 0.7f, 'u');
                BBoxRegression(rnet_res);
                BBoxPadSquare(rnet_res, image.cols, image.rows);

            }
            if (stage >= 3 && rnet_res.size()>0){
                int num = (int)rnet_res.size();
                int size = (int)ceil(1.f*num / StepSize);
                for (int iter = 0; iter < size; ++iter){
                    int start = iter*StepSize;
                    int end = min(start + StepSize, num);
                    vector<FaceInfo> input(rnet_res.begin() + start, rnet_res.begin() + end);
                    vector<FaceInfo> res = NextStage(image, input, 48, 48, 3, threshold[2]);
                    onet_res.insert(onet_res.end(), res.begin(), res.end());
                }
                BBoxRegression(onet_res);
                onet_res = NMS(onet_res, 0.7f, 'm');
                BBoxPad(onet_res, image.cols, image.rows);

            }
            if (stage == 1){
                return pnet_res;
            }
            else if (stage == 2){
                return rnet_res;
            }
            else if (stage == 3){
                return onet_res;
            }
            else{
                return onet_res;
            }
        }









    }
}