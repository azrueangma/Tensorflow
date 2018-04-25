from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import os, gzip

mnist = input_data.read_data_sets("./data/mnist/", one_hot = True)
def load_mnist():
    data_dir = os.path.join("./data/mnist")
    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trainX = data.reshape((60000, 28, 28, 1))
    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trainY = np.asarray(data.reshape((60000)))
    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    testX = data.reshape((10000, 28, 28, 1))
    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    testY = np.asarray(data.reshape((10000)))
    X = np.concatenate((trainX, testX), axis=0)
    Y = np.concatenate((trainY, testY), axis=0)
    np.random.seed(0)
    maskX = np.random.permutation(len(X))
    maskY = np.random.permutation(len(Y))
    X = X[maskX]
    Y = Y[maskY]
    Y_vec = np.zeros((len(Y), 10), dtype=np.float)
    for i, label in enumerate(Y):
        idx = int(i)
        Y_vec[idx, int(Y[idx])] = 1.0
    return X / 255., Y_vec
