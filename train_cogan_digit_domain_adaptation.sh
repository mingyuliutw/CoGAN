cd cogan/data;
tar xvf usps_digits;
tar xvf mnist_digits;
cd ../scripts;
mkdir ../logs/mnist2usps;
mkdir ../snapshots/mnist2usps;
../../caffe/build/tools/caffe_cogan.bin train --solver ../models/mnist2usps/solver.ptt;