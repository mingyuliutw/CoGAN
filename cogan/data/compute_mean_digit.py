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
import numpy as np
import cv2
import os
import sys
from os import path
from optparse import OptionParser;

parser = OptionParser();
parser.add_option('--folder', type=str, help="folder name"); 
parser.add_option('--list', type=str, help="list name"); 
parser.add_option('--dim', type=int, help="image dimension"); 
parser.add_option('--output', type=str, help="output image name"); 

(opts, args) = parser.parse_args(sys.argv)  
lines = [line.rstrip('\n') for line in open(opts.list)]

sumImg = np.zeros(shape=(opts.dim,opts.dim),dtype=np.float32)
count = 0
for l in lines:
	keys = l.split(' ')
	filename = path.join(opts.folder,keys[0])
	print filename
	img = cv2.imread(filename,0)
	sumImg += np.float32(img)
	count += 1
print count
output_img = np.uint8(sumImg*255.0/count)
cv2.imwrite(opts.output,output_img)


