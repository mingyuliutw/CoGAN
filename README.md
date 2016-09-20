# Coupled Generative Adversarial Network code

# General

This is an implementation of the Coupled Generative Adversarial Netowork algorithm. For more details please refer to our NIPS paper.

Please cite CoGAN in your publications if it helps your research:

@inproceedings{liu2016coupled,
  title={Coupled Generative Adversarial Networks},
  author={Liu, Ming-Yu and Tuzel, Oncel},
  booktitle={NIPS},
  year={2016}
}

# Content

/caffe: 

	The folder contains a modified version of Caffe. It includes a 	new solver and several new layers for supporting back propagation
	 with alternating gradient update steps. Note that several existing classes were modified for the same need.

/cogan:
	
	The folder contains the solver and network definition files.


#####################################################################
# Usage
#####################################################################

./build.sh
./train_cogan.sh
./test_cogan.sh

#####################################################################
# Copyright 2016, Ming-Yu Liu
#####################################################################

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
