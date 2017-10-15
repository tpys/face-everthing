/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Detection module, containing codes implementing the
 * face detection method described in the following paper:
 *
 *
 *   Funnel-structured cascade for multi-view face detection with alignment awareness,
 *   Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
 *   In Neurocomputing (under review)
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Shuzhe Wu (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#ifndef SEETA_FD_FUST_H_
#define SEETA_FD_FUST_H_

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "classifier.h"
#include "feature_map.h"
#include "model_reader.h"
#include "util/image_pyramid.h"

#include "cl_common.h"
#include "cl_detector.h"

namespace seeta {
namespace fd {

class SeetaDetector : public cl::fd::Detector {
 public:
    explicit SeetaDetector(int wnd_size = 40, int min_face_size = 20, int slide_wnd_step_x = 4, int slide_wnd_step_y = 4, float cls_thresh = 3.85, float scale_factor = 0.5)
      : wnd_size_(wnd_size),
        min_face_size_(min_face_size),
        slide_wnd_step_x_(slide_wnd_step_x),
        slide_wnd_step_y_(slide_wnd_step_y),
        cls_thresh_(cls_thresh),
        scale_factor_(scale_factor),
        num_hierarchy_(0) {
    wnd_data_buf_.resize(wnd_size_ * wnd_size_);
    wnd_data_.resize(wnd_size_ * wnd_size_);
    set_param(min_face_size_, slide_wnd_step_x_, slide_wnd_step_y_, cls_thresh_, scale_factor);
  }

  ~SeetaDetector() {}
  virtual bool load_model(const std::vector<std::string>& init_nets,
                            const std::vector<std::string>& predict_nets);
  virtual bool detect(const cv::Mat& src, std::vector<cl::FaceBox>& faces);


 private:
    void set_param(int min_face_size, int step_x, int step_y, float score_th, float scale_factor);

    inline void SetWindowSize(int32_t size) {
      if (size >= 20)
        wnd_size_ = size;
    }

    inline void SetSlideWindowStep(int32_t step_x, int32_t step_y) {
      if (step_x > 0)
        slide_wnd_step_x_ = step_x;
      if (step_y > 0)
        slide_wnd_step_y_ = step_y;
    }

 private:

  std::shared_ptr<seeta::fd::ModelReader> CreateModelReader(seeta::fd::ClassifierType type);
  std::shared_ptr<seeta::fd::Classifier> CreateClassifier(seeta::fd::ClassifierType type);
  std::shared_ptr<seeta::fd::FeatureMap> CreateFeatureMap(seeta::fd::ClassifierType type);

  void GetWindowData(const seeta::ImageData & img, const cv::Rect & wnd);

  int32_t wnd_size_;
  int32_t min_face_size_;
  int32_t slide_wnd_step_x_;
  int32_t slide_wnd_step_y_;
  float scale_factor_;
  float cls_thresh_;
  int32_t num_hierarchy_;

  std::vector<int32_t> hierarchy_size_;
  std::vector<int32_t> num_stage_;
  std::vector<std::vector<int32_t> > wnd_src_id_;

  std::vector<uint8_t> wnd_data_buf_;
  std::vector<uint8_t> wnd_data_;

  std::vector<std::shared_ptr<seeta::fd::Classifier> > model_;
  std::vector<std::shared_ptr<seeta::fd::FeatureMap> > feat_map_;
  std::map<seeta::fd::ClassifierType, int32_t> cls2feat_idx_;

  ImagePyramid img_pyramid_;

DISABLE_COPY_AND_ASSIGN(SeetaDetector);
};

}  // namespace fd
}  // namespace seeta

#endif  // SEETA_FD_FUST_H_
