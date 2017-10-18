#ifndef CL_FACE_NET_CAFFE_H
#define CL_FACE_NET_CAFFE_H

#include <array>
#include <memory> // std::shared_ptr
#include <string>
#include <vector>
#include <caffe/net.hpp>
#include "cl_common.h"

namespace cl
{
    class NetCaffe
    {
    public:
        NetCaffe(const std::string& init_net,
        const std::string& predict_net,
        const std::string& layer_name = "embedding",
        const int gpu_id = 0);

        virtual ~NetCaffe(){}

        bool init(int batch_size, int channels, int height, int width);
        void forward_pass(const float* const input_data = nullptr) const;
        void extract_feature(const int src_image_num, std::vector<std::vector<float>>& features) const;

        std::vector<float> extract_feature() const
        {
            const float* embedding = output_->cpu_data();
            int step = output_->num();
            std::vector<float> feature(embedding, embedding + step*output_->channels());
            return feature;
        }


    private:

        const int gpu_id_;
        const std::string init_net_;
        const std::string predict_net_;
        const std::string layer_name_;

        unsigned long input_memory_;

        std::unique_ptr<caffe::Net<float>> net_;
        boost::shared_ptr<caffe::Blob<float>> output_;

        DISABLE_COPY_AND_ASSIGN(NetCaffe);
    };
}

#endif // CL_FACE_NET_CAFFE_H
