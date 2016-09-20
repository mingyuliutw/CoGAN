mkdir build;
cd build;
cmake ../caffe/.;
make all -j8;
make pycaffe;
cp -r ../caffe/python/caffe ../cogan/tools/
