# Coupled Generative Adversarial Network code

# General

This is an implementation of the Coupled Generative Adversarial Netowork algorithm. For more details please refer to our NIPS paper.

Please cite the CoGAN paper in your publications if it is the code is useful to your research.

Also, a PyTorch implementation of CoGAN is available in https://github.com/mingyuliutw/CoGAN_PyTorch.

# Content

caffe/

The folder contains a modified version of Caffe. It includes a 	new solver and several new layers for supporting back propagation with alternating gradient update steps. Note that several existing classes were modified for the same need.

cogan/
	
The folder contains the solver and network definition files.

build.sh

A script for building the Caffe library.

train_cogan.sh

A script for train a CoGAN net.

test_cogan.sh

A script for visualizing CoGAN training results.


# Usage

./build.sh
./train_cogan.sh
./test_cogan.sh

# Copyright 2016, Ming-Yu Liu

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
