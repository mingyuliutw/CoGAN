#!/usr/bin/env python
import numpy as np
import os
import sys
import glob
import time
import sys
sys.path.append('../../caffe/python/')

import caffe
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from optparse import OptionParser;

parser = OptionParser();
parser.add_option('--model', type=str, help="network"); 
parser.add_option('--weights', type=str, help="trained weights"); 
parser.add_option('--output', type=str, help="output_image_name");
parser.add_option('--gpu', type=int, help="gpu device id");


def main(argv):
    (opts, args) = parser.parse_args(argv)  
    global model,mean,weights
    try:
        opts.model;
    except Exception, e:
        parser.print_usage();
        return -1;
    classifier = caffe.Classifier(opts.model, opts.weights)
    caffe.set_mode_gpu()    
    caffe.set_device(opts.gpu)
    fig=plt.figure(1)    
    idx = 1
    np.random.seed(0)
    # ins = np.random.normal(0, 1, (100,100,1,1))
    nsamples = 25
    gaps = 5
    ins = np.random.uniform(-1, 1, (nsamples,100,1,1))
    xspace = np.linspace(-1,1,gaps+1,endpoint=0)[1:]
    yspace = np.linspace(-1,1,gaps+1,endpoint=0)[1:]
    siz = len(xspace)
    caffe_in = ins   
    predictions0,predictions1 = classifier.render2(caffe_in)
    print predictions0.shape
    print predictions1.shape
    xdims = predictions0.shape[3]
    ydims = predictions0.shape[2]
    if(predictions1.shape[1]==1):
        for ix,x in enumerate(xspace):
            for iy,y in enumerate(yspace):
                img0 = np.zeros((ydims,xdims,3));
                img0[:,:,0] = predictions0[idx-1,2,:,:]*0.5+0.5
                img0[:,:,1] = predictions0[idx-1,1,:,:]*0.5+0.5
                img0[:,:,2] = predictions0[idx-1,0,:,:]*0.5+0.5

                img1 = np.zeros((ydims,xdims));
                img1[:,:] = predictions1[idx-1,0,:,:]
                plt.subplot(siz,2*siz,2*idx-1)
                plt.imshow(img0)
                plt.axis('off')
                plt.subplot(siz,2*siz,2*idx)
                plt.imshow(img1,cmap = cm.Greys_r)
                plt.axis('off')                                          
                idx += 1
    elif (predictions1.shape[1]==3):
        for ix,x in enumerate(xspace):
            for iy,y in enumerate(yspace):
                img0 = np.zeros((ydims,xdims,3));
                img0[:,:,0] = predictions0[idx-1,2,:,:]*0.5+0.5
                img0[:,:,1] = predictions0[idx-1,1,:,:]*0.5+0.5
                img0[:,:,2] = predictions0[idx-1,0,:,:]*0.5+0.5

                img1 = np.zeros((ydims,xdims,3));
                img1[:,:,0] = predictions1[idx-1,2,:,:]*0.5+0.5
                img1[:,:,1] = predictions1[idx-1,1,:,:]*0.5+0.5
                img1[:,:,2] = predictions1[idx-1,0,:,:]*0.5+0.5
                plt.subplot(siz,2*siz,2*idx-1)
                plt.imshow(img0)
                plt.axis('off')
                plt.subplot(siz,2*siz,2*idx)
                plt.imshow(img1)
                plt.axis('off')                                          
                idx += 1              
    try:
        print opts.output
        plt.savefig(opts.output,dpi=300,format="png")
    except:
        plt.show()
        return
if __name__ == '__main__':
    main(sys.argv)
