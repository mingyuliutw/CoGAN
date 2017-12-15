#ifndef CAFFE_RANDOM_NOISE_LAYERS_HPP_
#define CAFFE_RANDOM_NOISE_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
//====================================================
// Self defined layers
//====================================================

  template <typename Dtype>
  class RandomNoiseLayer : public Layer<Dtype> {
   public:
    explicit RandomNoiseLayer(const LayerParameter& param) : Layer<Dtype>(param) {}  
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "RandomNoise"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
   protected:    
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    int noise_type_;
    Blob<Dtype> rnd_vals_;
    Dtype std_;
    Dtype lb_;
    Dtype ub_;
  };
}  // namespace caffe

#endif  // CAFFE_RANDOM_NOISE_LAYERS_HPP_
