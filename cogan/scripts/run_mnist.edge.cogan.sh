#!/usr/bin/env bash
gpu=$1
exp=mnist.edge.cogan
for str in models snapshots logs; 
do
	if [ ! -d ../${str}/${exp} ]; then
	  mkdir -p ../${str}/${exp}
	fi
done

# Create the network definition files for training and testing
../tools/make_${exp}_net.py --snapshot_prefix ../snapshots/${exp}/${exp} --model_prefix ../models/${exp}/${exp} --data_header ../models/${exp}/${exp}.data.ptt --weight_filter msra --num 128 --dims 100 --g_neurons 1024,512,256,128,1 --g_kernels 4,3,3,3,6 --g_strides 0,2,2,2,1 --g_pads 0,1,1,1,1 --d_neurons 20,50,500 --d_kernels 5,5,0 --d_strides 1,1,0 --d_pads 0,0,0

# Perform training
../..//build/tools/caffe_cogan.bin train --solver ../models/${exp}/${exp}.solver.ptt --gpu ${gpu}