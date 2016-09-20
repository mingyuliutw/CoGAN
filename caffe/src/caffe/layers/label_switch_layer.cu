#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/label_switch_layer.hpp"

namespace caffe {


	template <typename Dtype>
	void LabelSwitchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const int iter = count_%update_base_;
		if( iter == update_bin_) {
			caffe_gpu_set<Dtype>(top[0]->count(),0,top[0]->mutable_gpu_data());
		}	else {
			caffe_copy<Dtype>(top[0]->count(),bottom[0]->mutable_gpu_data(),top[0]->mutable_gpu_data());
		}
		count_++;
	}

	template <typename Dtype>
	void LabelSwitchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		// caffe_gpu_set<Dtype>(bottom[0]->count(),0,bottom[0]->mutable_gpu_diff());
	}

INSTANTIATE_LAYER_GPU_FUNCS(LabelSwitchLayer);
}  // namespace caffe
