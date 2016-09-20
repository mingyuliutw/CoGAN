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

def separator_bar(fp,output_name):
	fp.write("#####################################################\n")
	fp.write("### %s\n" % output_name)
	fp.write("#####################################################\n")
	return

def copy_data_header(fp,header_filename):
	with open(header_filename) as f:
		content = f.readlines()	
	for l in content:
		fp.write(l)	
	fp.write('\n')

def dummy_data_layer(fp,output_name,num,dims,height=1,width=1):
	fp.write("layer { \n")
	fp.write("  top: \"%s\" name: \"%s\"\n" % (output_name,output_name) )
	fp.write("  type: \"DummyData\"\n")
	fp.write("  dummy_data_param { num: %d channels: %d height: %d width: %d}\n" % (num,dims,height,width))
	fp.write("}\n")
# 
# Noise layer
# 
def gaussian_noise_layer(fp,input_name,output_name,std_dev=1.0):
	fp.write("layer {\n")
	fp.write("  bottom: \"%s\" top: \"%s\" name: \"%s\"\n" % (input_name,output_name,output_name) )
	fp.write("  type: \"RandomNoise\"\n")
	fp.write("  random_noise_param { noise_type: GAUSSIAN std_dev: %f}\n" % std_dev)
	fp.write("}\n")
	return

def uniform_noise_layer(fp,input_name,output_name,lb=-1.0,ub=1.0):
	fp.write("layer {\n")
	fp.write("  bottom: \"%s\" top: \"%s\" name: \"%s\"\n" % (input_name,output_name,output_name) )
	fp.write("  type: \"RandomNoise\"\n")
	fp.write("  random_noise_param { noise_type: UNIFORM lb: %f ub: %f}\n" % (lb,ub))
	fp.write("}\n")
	return

def first_deconvolution_layer(fp,update_base,update_bin,input_name,output_name,channels,kernel_size,weight_filter):
	num_output = (channels*kernel_size*kernel_size)
	fp.write("layer {\n")
	fp.write("  bottom: \"%s\"  top: \"%s_ip\" name: \"%s_ip\" \n" % (input_name,output_name,output_name) )
	fp.write("  type: \"InnerProduct\"  \n")
	fp.write("  param { lr_mult: 1 decay_mult: 1 update_base: %d update_bin: %d}\n" %(update_base,update_bin) )
	fp.write("  param { lr_mult: 1 decay_mult: 0 update_base: %d update_bin: %d}\n" %(update_base,update_bin) )
	fp.write("  inner_product_param {\n")
	fp.write("    num_output: %d\n" % num_output)
	if(weight_filter=="msra"):
		fp.write("		weight_filler { type: \"msra\"}\n")
	else:
		fp.write("		weight_filler { type: \"gaussian\" std: 0.02}\n")
	fp.write("		bias_filler { type: \"constant\"}	\n")	
	fp.write("  }\n")
	fp.write("}\n")
	fp.write("layer {\n")
	fp.write("  bottom: \"%s_ip\" top: \"%s\" name: \"%s_reshape\" \n" % (output_name,output_name,output_name) )
	fp.write("  type: \"Reshape\"\n")
	fp.write("  reshape_param { shape { dim: 0 dim: %d dim: %d dim: %d}}\n" % (channels,kernel_size,kernel_size))
	fp.write("}	\n")
	return

def reshape_layer(fp,input_name,output_name,channels,height,width):
	fp.write("layer {\n")
	fp.write("  bottom: \"%s\" top: \"%s\" name: \"%s_reshape\" \n" % (input_name,output_name,output_name) )
	fp.write("  type: \"Reshape\"\n")
	fp.write("  reshape_param { shape { dim: 0 dim: %d dim: %d dim: %d}}\n" % (channels,height,width))
	fp.write("}	\n")
	return

def convolution_layer(fp,update_base,update_bin,input_name,output_name,num_output,kernel_size,pad_size,stride_length,weight_filter):
	fp.write("layer {\n")
	fp.write("  bottom: \"%s\" top: \"%s\" name: \"%s_conv\"\n" % (input_name,output_name,output_name))
	fp.write("  type: \"Convolution\"\n")
	fp.write("  param { lr_mult: 1 decay_mult: 1 update_base: %d update_bin: %d}\n" %(update_base,update_bin) );
	fp.write("  param { lr_mult: 1 decay_mult: 0 update_base: %d update_bin: %d}\n" %(update_base,update_bin) );
	fp.write("  convolution_param {\n");
	fp.write("    num_output: %d\n" % num_output)
	fp.write("    kernel_size: %d \n" % kernel_size)	
	fp.write("    pad: %d \n" % pad_size)
	fp.write("    stride: %d \n" % stride_length)
	if(weight_filter=="msra"):
		fp.write("		weight_filler { type: \"msra\"}\n")
	else:
		fp.write("		weight_filler { type: \"gaussian\" std: 0.02}\n")
	fp.write("		bias_filler { type: \"constant\" }	\n")	
	fp.write("  }\n")
	fp.write("}\n")
	return

def deconvolution_layer(fp,update_base,update_bin,input_name,output_name,num_output,kernel_size,pad_size,stride_length,weight_filter):
	fp.write("layer {\n")
	fp.write("  bottom: \"%s\" top: \"%s\" name: \"%s_conv\" \n" % (input_name,output_name,output_name) )
	fp.write("  type: \"Deconvolution\"\n")
	fp.write("  param { lr_mult: 1 decay_mult: 1 update_base: %d update_bin: %d}\n" %(update_base,update_bin) );
	fp.write("  param { lr_mult: 1 decay_mult: 0 update_base: %d update_bin: %d}\n" %(update_base,update_bin) );
	fp.write("  convolution_param {\n");
	fp.write("    num_output: %d\n" % num_output)
	fp.write("    kernel_size: %d\n" % kernel_size)	
	fp.write("    pad: %d\n" % pad_size)
	fp.write("    stride: %d\n" % stride_length)
	if(weight_filter=="msra"):
		fp.write("		weight_filler { type: \"msra\"}\n")
	else:
		fp.write("		weight_filler { type: \"gaussian\" std: 0.02}\n")
	fp.write("		bias_filler { type: \"constant\"}	\n")	
	fp.write("  }\n")
	fp.write("}\n")
	return

def inner_product_layer(fp,update_base,update_bin,input_name,output_name,num_neurons,weight_filter):
	fp.write("layer {\n")
	fp.write("  bottom: \"%s\" top: \"%s\" name: \"%s\"\n" % (input_name,output_name,output_name))
	fp.write("  type: \"InnerProduct\"\n")
	fp.write("  param { lr_mult: 1 decay_mult: 1 update_base: %d update_bin: %d}\n" %(update_base,update_bin) )
	fp.write("  param { lr_mult: 2 decay_mult: 0 update_base: %d update_bin: %d}\n" %(update_base,update_bin) )
	fp.write("  inner_product_param {\n")
	fp.write("    num_output: %d\n" % num_neurons)
	if(weight_filter=="msra"):
		fp.write("		weight_filler { type: \"msra\"}\n")
	else:
		fp.write("		weight_filler { type: \"gaussian\" std: 0.02}\n")
	fp.write("    bias_filler { type: \"constant\" }\n")
	fp.write("  }\n")
	fp.write("}\n")

def pooling_layer(fp,input_name,output_name,kernel_size,stride_size):
	fp.write("layer {\n")
	fp.write("  bottom: \"%s\" top: \"%s\" name: \"%s\"\n" % (input_name,output_name,output_name) )
	fp.write("  type: \"Pooling\"\n")
	fp.write("  pooling_param { pool: MAX kernel_size: %d stride: %d }\n" %(kernel_size,stride_size))
	fp.write("}	\n")
	return	

def dropout_layer(fp,layer_name,ratio=0.5):
	fp.write("layer { name: \"%s_relu\" type: \"Dropout\"  bottom: \"%s\" top: \"%s\" dropout_param { dropout_ratio: %f}}\n" %( layer_name,layer_name,layer_name,ratio) )
	return

def batch_norm_layer(fp,update_base,update_bin,output_name,use_global_stats=False):
	fp.write("layer { \n")
	fp.write("  bottom:\"%s\" top: \"%s\" name: \"%s_batchnorm\" \n" % (output_name,output_name,output_name) )
	fp.write("  type:\"BatchNorm\"\n")
	fp.write("	param { lr_mult: 1 decay_mult: 0 update_base: %d update_bin: %d}\n" %(update_base,update_bin) )
	fp.write("	param { lr_mult: 1 decay_mult: 0 update_base: %d update_bin: %d}\n" %(update_base,update_bin) )
	fp.write("	param { lr_mult: 1 decay_mult: 0 update_base: %d update_bin: %d}\n" %(update_base,update_bin) )
	if(use_global_stats==True):
		fp.write("	batch_norm_param {\n")
		fp.write("		use_global_stats: true\n")
		fp.write("	}	\n")
	fp.write("}	\n")
	return

def bias_layer(fp,update_base,update_bin,output_name):
	fp.write("layer {\n")
	fp.write("  bottom:\"%s\" top: \"%s\" name: \"%s_bias\" \n" % (output_name,output_name,output_name) )
	fp.write("	type: \"Bias\"\n")
	fp.write("  param { lr_mult: 1 decay_mult: 0 update_base: %d update_bin: %d}\n" %(update_base,update_bin) )
	fp.write("}	\n")
	return	

def prelu_layer(fp,update_base,update_bin,output_name):
	fp.write("layer { \n")
	fp.write("  bottom:\"%s\" top: \"%s\" name: \"%s_prelu\" \n" % (output_name,output_name,output_name) )
	fp.write("  type:\"PReLU\"\n")
	fp.write("  param { lr_mult: 1 decay_mult: 0 update_base: %d update_bin: %d}\n" %(update_base,update_bin) )
	fp.write("}\n")
	return	

def sigmoid_layer(fp,output_name):
	fp.write("layer { name: \"%s_sigmoid\" bottom: \"%s\" top: \"%s\" type: \"Sigmoid\"}\n" %( output_name,output_name,output_name) )
	return

def tanh_layer(fp,output_name):
	fp.write("layer { name: \"%s_tanh\" bottom: \"%s\" top: \"%s\" type: \"TanH\"}\n" %( output_name,output_name,output_name) )
	return

def groundtruth_label_layer(fp,update_base,update_bin,output_name,num_samples):
	fp.write("layer { \n")
	fp.write("  top: \"true_%s\" name: \"true_%s\" \n" % (output_name,output_name) )
	fp.write("  type: \"DummyData\" \n")
	fp.write("  dummy_data_param { num: %d channels: 1 height: 1 width: 1 data_filler { value: 1 }}\n" % num_samples)
	fp.write("}\n")
	fp.write("layer {\n")
	fp.write("  bottom: \"true_%s\" top: \"fake_%s\" name: \"fake_%s\"\n" % (output_name,output_name,output_name) )
	fp.write("  type: \"LabelSwitch\"\n")
	fp.write("  label_switch_param { update_base: %d update_bin: %d}\n" %(update_base,update_bin) )
	fp.write("}\n")
	concate_layer(fp,"true_%s" % output_name,"fake_%s" % output_name,output_name,output_name)
	return

def concate_layer(fp,input_name1,input_name2,output_name,layer_name):	
	fp.write("layer {\n")
	fp.write("  bottom: \"%s\" bottom: \"%s\" top: \"%s\" name: \"%s\" \n" % (input_name1,input_name2,output_name,layer_name) )
	fp.write("  type: \"Concat\"\n")
	fp.write("  concat_param { axis: 0 }\n")
	fp.write("}	\n")
	return	

def discriminator_output_layer(fp,update_base,update_bin,input_name,output_name,label_name,weight_filter):	
	inner_product_layer(fp,update_base,update_bin,input_name,output_name,2,weight_filter)
	fp.write("layer {\n")
	fp.write("  bottom: \"%s\" bottom: \"%s\" top: \"loss\" name: \"loss\"\n" % (output_name,label_name))
	fp.write("  type: \"SoftmaxWithLoss\"\n")	
	fp.write("  propagate_down: 1\n")
	fp.write("  propagate_down: 0\n")
	fp.write("}\n")
	fp.write("layer {\n")
	fp.write("  bottom: \"%s\" bottom: \"%s\" top: \"accuracy\" name: \"accuracy\"\n" % (output_name,label_name))
	fp.write("  type: \"Accuracy\"\n")
	fp.write("}\n")
	return

def input_layer(fp,output_name,num,channels,height,width):
	fp.write("layer {\n")
	fp.write("	top: \"%s\" name: \"%s\"\n" % (output_name,output_name))
	fp.write("	type: \"Input\"\n")
	fp.write("	input_param { shape: { dim: %d dim: %d dim: %d dim: %d } }\n" %(num,channels,height,width))
	fp.write("}	\n")