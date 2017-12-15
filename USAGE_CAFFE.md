# Coupled Generative Adversarial Network code
## Using the repository with the [Caffe](https://github.com/BVLC/caffe) library

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
