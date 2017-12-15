# Coupled Generative Adversarial Network code

## General

This is the open source repository for the Coupled Generative Adversarial Network (CoupledGAN or CoGAN) work.  For more details please refer to our [NIPS 2016 paper](https://papers.nips.cc/paper/6544-coupled-generative-adversarial-networks.pdf) or our [arXiv paper](https://arxiv.org/abs/1606.07536). Please cite the NIPS paper in your publications if you find the source code useful to your research.

I have improved the algorithm by combining with encoders. For more details please check our [NIPS 2017 paper on Unsupervised Image-to-Image Translation Networks](https://papers.nips.cc/paper/6672-unsupervised-image-to-image-translation-networks.pdf)

## USAGE

In this repository, we provide both [Caffe](https://github.com/BVLC/caffe) implementation and [PyTorch](http://pytorch.org/) implementation. For using the code with the [Caffe](https://github.com/BVLC/caffe) library, please consult [USAGE_CAFFE](USAGE_CAFFE.md). For using the code with the [PyTorch](http://pytorch.org/) library, please consult [USAGE_PYTORCH](USAGE_PYTORCH.md).


## CoGAN Network Architecture
![](https://github.com/mingyuliutw/CoGAN_PyTorch/blob/master/images/overview_landscape_very_tight.jpg)

## CoGAN learn to generate corresponding smile and non-smile faces
![](https://github.com/mingyuliutw/CoGAN_PyTorch/blob/master/images/result_face_smiling_small.jpg)

## CoGAN learn to generate corresponding faces with blond-hair and without non-blond-hair
![](https://github.com/mingyuliutw/CoGAN_PyTorch/blob/master/images/result_face_blondhair_small.jpg)

## CoGAN learn to generate corresponding faces with eye-glasses and without eye-glasses
![](https://github.com/mingyuliutw/CoGAN_PyTorch/blob/master/images/result_face_eyeglasses_small.jpg)

## CoGAN learn to generate corresponding RGB and depth images
![](https://github.com/mingyuliutw/CoGAN_PyTorch/blob/master/images/result_nyu_small.jpg)

---

Copyright 2017, Ming-Yu Liu
All Rights Reserved

Permission to use, copy, modify, and distribute this software and its documentation for any non-commercial purpose is hereby granted without fee, provided that the above copyright notice appear in all copies and that both that copyright notice and this permission notice appear in supporting documentation, and that the name of the author not be used in advertising or publicity pertaining to distribution of the software without specific, written prior permission.

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.