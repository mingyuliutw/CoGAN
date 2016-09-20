cd cogan/scripts/;
mkdir ../data;
mkdir ../data/mnist.edge.cogan;
wget http://deeplearning.net/data/mnist/mnist.pkl.gz;
mv *.gz ../data
../tools/prepare_mnist.edge.cogan.py;
./run_mnist.edge.cogan.sh 0