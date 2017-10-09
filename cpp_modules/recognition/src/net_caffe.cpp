#include "net_caffe.h"

#include <numeric> // std::accumulate
#include <cuda.h>
#include <cuda_runtime.h>


namespace cl
{
    using namespace cv;
    using namespace std;

    NetCaffe::NetCaffe(const std::string& init_net,
     const std::string& predict_net,
     const std::string& layer_name,
     const int gpu_id):
        gpu_id_{gpu_id},
        init_net_{init_net},
        predict_net_{predict_net},
        layer_name_{layer_name}
    {
    }

    bool NetCaffe::init(int batch_size, int channels, int height, int width)
    {
        input_memory_ = batch_size*channels*height*width*sizeof(float);
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        caffe::Caffe::SetDevice(gpu_id_);
        net_.reset(new caffe::Net<float>{init_net_, caffe::TEST});
        net_->CopyTrainedLayersFrom(predict_net_);
        net_->blobs()[0]->Reshape({batch_size, channels, height, width});
        net_->Reshape();
        output_ = net_->blob_by_name(layer_name_);
        return true;
    }


    void NetCaffe::forward_pass(const float* const input_data) const
    {
        if (input_data != nullptr)
        {
            auto* input_data_gpu = net_->blobs().at(0)->mutable_gpu_data();
            cudaMemcpy(input_data_gpu, input_data, input_memory_, cudaMemcpyHostToDevice);
        }
        net_->ForwardFrom(0);
    }

    void NetCaffe::extract_feature(const int src_image_num, std::vector<std::vector<float>>& features) const
     {
         const float* embedding = output_->cpu_data();
         features.resize(src_image_num);
         int step = output_->num()/src_image_num;
         int offset = 0;
         for(int i = 0; i < src_image_num; ++i){
             features[i] = vector<float>(embedding + offset, embedding + offset + step*output_->channels());
             offset += step*output_->channels();
         }
     }



}
