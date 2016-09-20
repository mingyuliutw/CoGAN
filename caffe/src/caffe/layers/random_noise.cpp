#include <algorithm>
#include <vector>
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/random_noise_layer.hpp"

namespace caffe {

template <typename Dtype>
void RandomNoiseLayer<Dtype>::LayerSetUp( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {	
	noise_type_ = (this->layer_param_.random_noise_param().noise_type());	
	switch(noise_type_) { 
		case 0: // Uniform
			lb_ = (Dtype)(this->layer_param_.random_noise_param().lb());	
			ub_ = (Dtype)(this->layer_param_.random_noise_param().ub());	
		break;		
		case 1: // Gaussian
			std_ = (Dtype)(this->layer_param_.random_noise_param().std_dev());
			CHECK( std_ != 0);
		break;
		default:
			CHECK(0==1) <<"We do not support other noise types.";
		break;
	}
}

template <typename Dtype>
void RandomNoiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  rnd_vals_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RandomNoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int count = bottom[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  switch(noise_type_) {
  	case 0: //Uniform
		  caffe_rng_uniform<Dtype>(count, lb_, ub_, rnd_vals_.mutable_cpu_data() );
		  caffe_sub(count, bottom_data, rnd_vals_.cpu_data(), top_data); 
  	break;
  	case 1: //Gaussian
		  caffe_rng_gaussian<Dtype>(count, (Dtype)(0.0), std_, rnd_vals_.mutable_cpu_data() );
		  caffe_sub(count, bottom_data, rnd_vals_.cpu_data(), top_data); 
  	break;

  	default:
  	break;
  }
}

template <typename Dtype>
void RandomNoiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {	
	// Nothing to do
}

#ifdef CPU_ONLY
STUB_GPU(RandomNoiseLayer);
#endif

INSTANTIATE_CLASS(RandomNoiseLayer);
REGISTER_LAYER_CLASS(RandomNoise);

}  // namespace caffe
