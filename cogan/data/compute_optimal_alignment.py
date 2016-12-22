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
from sklearn.metrics import mean_squared_error

imgU = cv2.imread("usps_mean.png",0)
imgM = cv2.imread("mnist_mean.png",0)

for p in range(0,10):
	newU = np.zeros(shape=(16+2*p,16+2*p),dtype=np.float32)
	h = newU.shape[0]
	newU[p:h-p,p:h-p] = np.float32(imgU)
	resizedU = cv2.resize(newU,(28,28))
	resizedU - np.float32(imgM)
	error = mean_squared_error(resizedU, np.float32(imgM))
	print "%d: %f" %(p,error)

# The optimal padding size is 3 on each side