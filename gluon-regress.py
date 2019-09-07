from mxnet import nd,gluon,autograd
from mxnet.gluon import nn,loss as gloss
from mxnet.gluon import data as gdata

# samples and ground truth
samples = nd.random_normal(0, 1, shape=(100, 2))
print(samples[0:3])
true_w = nd.array([0.3, 0.5])
true_w.reshape(2, 1)

true_b = 10

labels = nd.dot(samples, true_w) + true_b

net = nn.Sequential()
net.add(nn.Dense(1))

net.initialize()

batchSize = 10
batch_iter = gdata.DataLoader(gdata.ArrayDataset(samples,labels),batchSize,shuffle=True)

loss = gloss.L2Loss()

trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})

def epoch():
    epoch_num = 5
    for i in range(epoch_num):
        for mini_batch,mini_label in batch_iter:
            with autograd.record():
                output = net.forward(mini_batch)
                loss_batch = loss(output, mini_label)
            loss_batch.backward()
            trainer.step(batchSize)
        loss_total = loss(net(samples),labels)
        print("epoche:%d loss:%f" % (i+1,loss_total.mean().asnumpy()))

epoch()