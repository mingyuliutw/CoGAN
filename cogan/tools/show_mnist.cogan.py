#!/usr/bin/env python
import numpy as np
import os
import sys
import caffe
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from optparse import OptionParser;
parser = OptionParser();
parser.add_option('--model', type=str, help="network"); 
parser.add_option('--weights', type=str, help="trained weights"); 
parser.add_option('--output', type=str, help="output_image_name");


def main(argv):
    (opts, args) = parser.parse_args(argv)  
    try:
        opts.model;
        opts.weights;        
    except Exception, e:
        parser.print_usage();
        return -1;
    classifier = caffe.Classifier(opts.model, opts.weights)
    caffe.set_mode_gpu()    
    fig=plt.figure(1)    
    np.random.seed(0)
    ins = np.random.normal(0, 1, (100,100,1,1))
    xspace = np.linspace(-1,1,11,endpoint=0)[1:]
    yspace = np.linspace(-1,1,11,endpoint=0)[1:]
    siz = len(xspace)
    caffe_in = ins   
    predictions0,predictions1 = classifier.render2(caffe_in)    

    idx = 1
    for ix,x in enumerate(xspace):
        for iy,y in enumerate(yspace):
            img0 = np.zeros((28,28));
            img1 = np.zeros((28,28));
            img0[:,:] = predictions0[idx-1,0,:,:]            
            img1[:,:] = predictions1[idx-1,0,:,:]
            plt.subplot(siz,2*siz,2*idx-1)
            plt.imshow(img0,cmap = cm.Greys_r)
            plt.axis('off')
            plt.subplot(siz,2*siz,2*idx)
            plt.imshow(img1,cmap = cm.Greys_r)
            plt.axis('off')                                          
            idx += 1    
    try:
        print opts.output
        plt.savefig(opts.output,dpi=400,format="png")
    except:
        plt.show()
        return

if __name__ == '__main__':
    main(sys.argv)
