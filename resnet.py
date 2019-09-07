from mxnet import nd,gluon,init
from mxnet.gluon import nn
import d2lzh as d2l

net = nn.Sequential()

net.add(nn.Conv2D(channels=64,kernel_size=7,strides=2,padding=3))
net.add(nn.BatchNorm())
net.add(nn.MaxPool2D(pool_size=3,strides=2,padding=1))

_X = nd.random.uniform(low=0,high=1,shape=(5,3,224,224)) #5张3通道的图片
print("temp test data shapte:",_X.shape)


class Residual(nn.Block):
    def __init__(self,num_channels,num_strides,use_1x1=False,**kwargs):
        super(Residual, self).__init__(**kwargs)
        
        self.cov1 = nn.Conv2D(channels=num_channels,kernel_size=3,strides=num_strides,padding=1) #1. out_w = (x_w - k + 2p +1)/s
        self.cov2 = nn.Conv2D(channels=num_channels,kernel_size=3,strides=1,padding=1       )
        if use_1x1:  # strip == 2
            assert(num_strides == 2)
            self.cov3=nn.Conv2D(channels=num_channels,kernel_size=1,strides=num_strides)  #2. out_hw= (x_w -k +1)/s    1.eq 2.
        else:
            self.cov3=None

        self.batch_norm1 = nn.BatchNorm()
        self.batch_norm2 = nn.BatchNorm()
        

    def forward(self,X):
        Y= nd.relu(self.batch_norm1(self.cov1(X)))
        Y= self.batch_norm2(self.cov2(Y))
        if self.cov3:
            X=self.cov3(X)

        return nd.relu(Y+X)

#for test
# res_block = Residual(num_channels=3,num_strides=1,use_1x1=False)
# res_block.initialize()    
# print("res_block shape", res_block(_X).shape)

# res_block = Residual(num_channels=6,num_strides=2,use_1x1=True)
# res_block.initialize()
# print("res_block shape with 1X1", res_block(_X).shape)


def residaul_seq(num_block,channels,firstConnect=False):
    seq = nn.Sequential()
    for i in range(num_block):
        if i==0 and firstConnect: #do not shrink the ouput
            seq.add(Residual(num_channels=channels,num_strides=1,use_1x1=False))
        elif i!=0 and firstConnect: #do not shrink
            seq.add(Residual(num_channels=channels,num_strides=1,use_1x1=False))
        elif i==0 and not firstConnect: #shrink
            seq.add(Residual(num_channels=channels,num_strides=2,use_1x1=True))
        elif i!=0 and not firstConnect: # do not shrink
            seq.add(Residual(num_channels=channels,num_strides=1,use_1x1=False))
    return seq

num_crs = 64 #number of channels for each residual_seq
net.add(residaul_seq(num_block=2,channels=num_crs,firstConnect=True))
net.add(residaul_seq(num_block=2,channels=2*num_crs))
net.add(residaul_seq(num_block=2,channels=4*num_crs))
net.add(residaul_seq(num_block=2,channels=8*num_crs))

#dense layers
net.add(nn.GlobalAvgPool2D())
net.add(nn.Dense(10))

net.initialize()


# for layer in net:
#     _X=layer(_X)
#     print('layer name:',layer.name," output shape",_X.shape)


#true config for net
lr=0.05
num_epochs=2
batch_size=256 #256
ctx=d2l.try_gpu()

net.initialize(force_reinit=True,ctx=ctx,init=init.Xavier())
trainer = gluon.Trainer(n
et.collect_params(),'sgd',{'learning_rate':lr})
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size,resize=96)
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,num_epochs)

# for X, y in train_iter:
#     X, y = X.as_in_context(ctx), y.as_in_context(ctx)
#     print('X shape',X.shape)
#     y_hat = net(X)
#     print('out shape',y_hat.shape)
#     break