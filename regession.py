from mxnet import nd, autograd

# samples and ground truth
samples = nd.random_normal(0, 1, shape=(100, 2))
print(samples[0:3])
true_w = nd.array([0.3, 0.5])
true_w.reshape(2, 1)

true_b = 10

labels = nd.dot(samples, true_w) + true_b
print(labels.shape)
assert labels.shape == (100,)

train_w = nd.random_normal(0, 1, shape=(2, 1))  # waiting for training
train_b = nd.ones(shape=(1,1))

train_w.attach_grad()
train_b.attach_grad()
# mini batch
def mini_batch(samples, labels, batchSize):
    sampleN = samples.shape[0]
    randomIndex = nd.array(list(range(sampleN)))
    nd.shuffle(randomIndex)

    for i in range(0, sampleN, batchSize):
        batchIndex = randomIndex[i : min(i + batchSize, sampleN)]
        yield samples.take(batchIndex), labels.take(batchIndex)


batchSize = 10
for x, y in mini_batch(samples, labels, batchSize):
    print("first batch")
    print(x, y)
    break


def network(batch):
    return nd.dot(batch, train_w) + train_b


def loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def optimizer(params, lr, batchSize):
    for param in params:
        param[:] = param - lr * param.grad / batchSize


# epoch
def epoch():
    rollNum = 30
    learningRate = 0.03
    batchSize = 10
    for i in range(rollNum):
        for batch, y in mini_batch(samples, labels, batchSize):
            with autograd.record():
                loss_batch = loss(network(batch), y)
            loss_batch.backward()
            optimizer([train_w, train_b], learningRate, batchSize)

        loss_total = loss(network(samples), labels)
        print("epoch：%d ,total_loss: %f" % (i + 1, loss_total.mean().asnumpy()))


epoch()
