#include <algorithm>
#include <vector>
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/random_noise_layer.hpp"

namespace caffe {

template <typename Dtype>
void RandomNoiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) { 
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int count = bottom[0]->count();   
  Dtype* top_data = top[0]->mutable_gpu_data();
  switch (noise_type_) {
    case 0: // Uniform nise
      caffe_gpu_rng_uniform<Dtype>(count, lb_, ub_, rnd_vals_.mutable_gpu_data() );
      caffe_gpu_sub(count, bottom_data, rnd_vals_.gpu_data(), top_data);      
    break;    
  	case 1: // Gaussian noise
			caffe_gpu_rng_gaussian<Dtype>(count, (Dtype)(0.0), std_, rnd_vals_.mutable_gpu_data() );
			caffe_gpu_sub(count, bottom_data, rnd_vals_.gpu_data(), top_data);
  	break;
  	default:
  	break;  	
  }
}

template <typename Dtype>
void RandomNoiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// Nothing to do.
}

INSTANTIATE_LAYER_GPU_FUNCS(RandomNoiseLayer);

}  // namespace caffe
