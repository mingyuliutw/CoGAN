#!/usr/bin/env python
'''
/* Copyright 2016, Ming-Yu Liu

All Rights Reserved 

Permission to use, copy, modify, and distribute this software and 
its documentation for any non-commercial purpose is hereby granted 
without fee, provided that the above copyright notice appear in 
all copies and that both that copyright notice and this permission 
notice appear in supporting documentation, and that the name of 
the author not be used in advertising or publicity pertaining to 
distribution of the software without specific, written prior 
permission. 

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, 
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
ANY PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES 
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN 
AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING 
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. 
*/
'''
import sys
import os
import numpy as np
from optparse import OptionParser;
import gan_lib

def array_callback(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))

parser = OptionParser();
parser.add_option('--data_header', type=str, help="data layer header file");
parser.add_option('--model_prefix', type=str, help="network name prefix");
parser.add_option('--snapshot_prefix', type=str, help="snapshot prefix");
parser.add_option('--weight_filter', type=str, help="weight filter type");
parser.add_option('--num', type=int, help="# of random noise samples");
parser.add_option('--dims', type=int, help="random noise dimensions");

parser.add_option('--g_neurons', type=str, action='callback', callback= array_callback, help="neuron size for the generator layers");
parser.add_option('--g_kernels', type=str, action='callback', callback= array_callback, help="kernel size for the generator layers");
parser.add_option('--g_strides', type=str, action='callback', callback= array_callback, help="stride size for the generator layers");
parser.add_option('--g_pads',    type=str, action='callback', callback= array_callback, help="pad size for the generator layers");

parser.add_option('--d_neurons', type=str, action='callback', callback= array_callback, help="neuron size for the discriminator layers");
parser.add_option('--d_kernels', type=str, action='callback', callback= array_callback, help="kernel size for the discriminator layers");
parser.add_option('--d_strides', type=str, action='callback', callback= array_callback, help="stride size for the discriminator layers");
parser.add_option('--d_pads',    type=str, action='callback', callback= array_callback, help="pad size for the discriminator layers");

def make_gan_solver(filename,opts):
	net_name = opts.model_prefix + ".train.ptt"	
	snapshot_name = opts.snapshot_prefix
	fp = open(filename,"w")
	fp.write("net: \"%s\"\n" % net_name)
	fp.write("base_lr: 0.0002\n")
	fp.write("momentum: 0.5\n")
	fp.write("momentum2: 0.999\n")
	fp.write("lr_policy: \"fixed\"\n")
	fp.write("type: \"AlterAdam\"\n")
	fp.write("weight_decay: 0.0005\n")
	fp.write("display: 10\n")
	fp.write("max_iter: 50000\n")
	fp.write("snapshot: 5000\n")
	fp.write("snapshot_prefix: \"%s\"\n" % snapshot_name)
	fp.write("solver_mode: GPU\n")
	fp.close()	
	return

def make_gan_train_net(filename,opts):
	fp = open(filename,"w")
	num = opts.num
	dims= opts.dims
	weight_filter = opts.weight_filter
	gan_lib.separator_bar(fp,"Data")
	gan_lib.copy_data_header(fp,opts.data_header)	

	gan_lib.separator_bar(fp,"Generator")
	update_base = 2
	update_bin  = 1	
	input_name  = "zero_values"
	output_name = "random_noise"	
	gan_lib.dummy_data_layer(fp,input_name,num,dims,1,1)	
	gan_lib.uniform_noise_layer(fp,input_name,output_name)
	nLayers= len(opts.g_neurons)
	for i in range(0,nLayers):
		input_name = output_name
		output_name = "dconv%d" % i		
		neuron = (int)(opts.g_neurons[i])
		kernel = (int)(opts.g_kernels[i])
		stride = (int)(opts.g_strides[i])
		pad 	 = (int)(opts.g_pads[i])		
		string = "%s: (n%d,k%d,s%d,p%d)" %(output_name,neuron,kernel,stride,pad)
		print string
		gan_lib.separator_bar(fp,string)
		if(i==0):
			gan_lib.first_deconvolution_layer(fp,update_base,update_bin,input_name,output_name,neuron,kernel,weight_filter)
			gan_lib.batch_norm_layer(fp,update_base,update_bin,output_name)
			gan_lib.bias_layer(fp,update_base,update_bin,output_name)
			gan_lib.prelu_layer(fp,update_base,update_bin,output_name)
		else:
			gan_lib.deconvolution_layer(fp,update_base,update_bin,input_name,output_name,neuron,kernel,pad,stride,weight_filter)
			if(i==nLayers-1):
				gan_lib.sigmoid_layer(fp,output_name)
			else:
				gan_lib.batch_norm_layer(fp,update_base,update_bin,output_name)
				gan_lib.bias_layer(fp,update_base,update_bin,output_name)
				gan_lib.prelu_layer(fp,update_base,update_bin,output_name)

	update_base = 2
	update_bin  = 0
	
	gan_lib.separator_bar(fp,"Data and ground truth labels")
	label_name = "labels"
	gan_lib.groundtruth_label_layer(fp,update_base,update_bin,label_name,num)
	input_name  = output_name
	output_name = "data_and_fake_data"
	gan_lib.concate_layer(fp,"data",input_name,output_name,output_name)

	gan_lib.separator_bar(fp,"Discriminator")

	nLayers= len(opts.d_neurons)	
	for i in range(0,nLayers):
		input_name = output_name
		output_name= "conv%d" % i
		neuron = (int)(opts.d_neurons[i])
		kernel = (int)(opts.d_kernels[i])
		stride = (int)(opts.d_strides[i])
		pad 	 = (int)(opts.d_pads[i])			
		string = "%s: (n%d,k%d,s%d,p%d)" %(output_name,neuron,kernel,stride,pad)
		print string
		gan_lib.separator_bar(fp,string)
		if(i==nLayers-1):
			gan_lib.inner_product_layer(fp,update_base,update_bin,input_name,output_name,neuron,weight_filter)
			gan_lib.prelu_layer(fp,update_base,update_bin,output_name)
			gan_lib.dropout_layer(fp,output_name)
		else:
			gan_lib.convolution_layer(fp,update_base,update_bin,input_name,output_name,neuron,kernel,pad,stride,weight_filter)
			input_name  = output_name
			output_name = "pool%d" % i		
			gan_lib.pooling_layer(fp,input_name,output_name,2,2)				

	gan_lib.separator_bar(fp,"Loss and Accuracy")
	score_output_name = "score_output"
	gan_lib.discriminator_output_layer(fp,update_base,update_bin,output_name,score_output_name,label_name,weight_filter)
	fp.close()
	return

def make_gan_deploy_net(filename,opts):
	fp = open(filename,"w")
	num = opts.num
	dims= opts.dims	
	update_base = 2
	update_bin  = 1		
	weight_filter = opts.weight_filter
	use_global_stats = True

	gan_lib.separator_bar(fp,"Generator")		
	output_name = "noise"
	gan_lib.input_layer(fp,output_name,100,dims,1,1)
	nLayers= len(opts.g_neurons)
	for i in range(0,nLayers):
		input_name = output_name
		output_name = "dconv%d" % i		
		neuron = (int)(opts.g_neurons[i])
		kernel = (int)(opts.g_kernels[i])
		stride = (int)(opts.g_strides[i])
		pad 	 = (int)(opts.g_pads[i])		
		string = "%s: (n%d,k%d,s%d,p%d)" %(output_name,neuron,kernel,stride,pad)
		print string
		gan_lib.separator_bar(fp,string)
		if(i==0):
			gan_lib.first_deconvolution_layer(fp,update_base,update_bin,input_name,output_name,neuron,kernel,weight_filter)
			gan_lib.batch_norm_layer(fp,update_base,update_bin,output_name,use_global_stats)
			gan_lib.bias_layer(fp,update_base,update_bin,output_name)
			gan_lib.prelu_layer(fp,update_base,update_bin,output_name)
		else:
			gan_lib.deconvolution_layer(fp,update_base,update_bin,input_name,output_name,neuron,kernel,pad,stride,weight_filter)
			if(i==nLayers-1):
				gan_lib.sigmoid_layer(fp,output_name)
			else:
				gan_lib.batch_norm_layer(fp,update_base,update_bin,output_name,use_global_stats)
				gan_lib.bias_layer(fp,update_base,update_bin,output_name)
				gan_lib.prelu_layer(fp,update_base,update_bin,output_name)

	return

def make_gan_net(argv):
	(opts, args) = parser.parse_args(argv)	
	try:
		opts.model_prefix
		opts.data_header
	except Exception, e:
	  parser.print_usage();
	  return -1;

	filename = opts.model_prefix + ".solver.ptt"
	make_gan_solver(filename,opts)	  

	filename = opts.model_prefix + ".train.ptt"
	make_gan_train_net(filename,opts)

	filename = opts.model_prefix + ".deploy.ptt"
	make_gan_deploy_net(filename,opts)

	return

if __name__ == '__main__':	
  make_gan_net(sys.argv)  