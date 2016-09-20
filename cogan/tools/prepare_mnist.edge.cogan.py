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
import cPickle, gzip
import numpy as np
import os
import lmdb
import shutil
import caffe
import cv2

binary_file_path = "../data/mnist.pkl.gz"
domain_1_lmdb = "../data/mnist.edge.cogan/mnist.edge.cogan.train.domain1.lmdb"
domain_2_lmdb = "../data/mnist.edge.cogan/mnist.edge.cogan.train.domain2.lmdb"


if os.path.isdir(domain_1_lmdb):
	shutil.rmtree(domain_1_lmdb)
if os.path.isdir(domain_2_lmdb):
	shutil.rmtree(domain_2_lmdb)

f = gzip.open(binary_file_path, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

data = np.uint8(255*np.concatenate((train_set[0],valid_set[0]),axis=0))
nSamples = data.shape[0]
arr = np.arange(nSamples)
np.random.shuffle(arr)

map_size = data.nbytes * 5

env1 = lmdb.open(domain_1_lmdb, map_size=map_size)
with env1.begin(write=True) as txn:
	for i in range(0,nSamples/2):
		datum = caffe.proto.caffe_pb2.Datum()
		datum.channels = 1
		datum.height = 28
		datum.width  = 28
		datum.data = data[arr[i],:].tobytes()
		datum.label = int(0)
		str_id = '{:08}'.format(i)
		txn.put(str_id.encode('ascii'), datum.SerializeToString())
print "Done with %s" % domain_1_lmdb

env2 = lmdb.open(domain_2_lmdb, map_size=map_size)    
with env2.begin(write=True) as txn:
  for i in range(nSamples/2,nSamples):
		datum = caffe.proto.caffe_pb2.Datum()
		datum.channels = 1
		datum.height = 28
		datum.width  = 28
		img = data[arr[i],:].reshape(28,28)
		dilation = cv2.dilate(img,np.ones((3,3),np.uint8),iterations = 1)    
		edge = dilation-img
		# plt.imshow(edge,cmap = cm.Greys_r)
		# plt.axis('off')               
		# plt.show()
		datum.data = edge.tobytes()
		datum.label = int(0)
		str_id = '{:08}'.format(i)
		txn.put(str_id.encode('ascii'), datum.SerializeToString())
print "Done with %s" % domain_2_lmdb