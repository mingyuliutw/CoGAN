#include <algorithm>
#include <vector>
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/label_switch_layer.hpp"

namespace caffe {

template <typename Dtype>
void LabelSwitchLayer<Dtype>::LayerSetUp( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {	
  update_base_ = this->layer_param().label_switch_param().update_base();
  update_bin_ = this->layer_param().label_switch_param().update_bin();
  count_ = 0;
}

template <typename Dtype>
void LabelSwitchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(),1,1,1); // label
}

template <typename Dtype>
void LabelSwitchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
	const int iter = count_%update_base_;
	if( iter == update_bin_) {
		caffe_set<Dtype>(top[0]->count(),0,top[0]->mutable_cpu_data());
	}	else {
		caffe_copy<Dtype>(top[0]->count(),bottom[0]->mutable_cpu_data(),top[0]->mutable_cpu_data());
	}
	count_++;
}

template <typename Dtype>
void LabelSwitchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {	

}

#ifdef CPU_ONLY
STUB_GPU(LabelSwitchLayer);
#endif

INSTANTIATE_CLASS(LabelSwitchLayer);
REGISTER_LAYER_CLASS(LabelSwitch);

}  // namespace caffe
