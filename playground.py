from mxnet import nd,gluon,autograd
from mxnet.gluon import nn,loss as gloss
from mxnet.gluon import data as gdata


assert(1 == 1)


assert(2 == int('2'))

sa = nd.random_normal(0,1,shape=(2,3))
print(sa.shape[0])
print(sa[2:])

mnist_train = gdata.vision.FashionMcNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)