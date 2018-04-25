from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import shutil
import os

def conv2d(x, kernel_width, kernel_height, kernel_channel, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name = 'W', shape = [kernel_width,  kernel_height, x.get_shape()[-1], kernel_channel], initializer = tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [kernel_channel], initializer = tf.zeros_initializer())
        h = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')+b,name = 'h')
    return h

def max_pool(x, pooling_width, pooling_height, nstride, name):
    with tf.variable_scope(name):
        p = tf.nn.max_pool(x, ksize = [1, pooling_width, pooling_height, 1], strides = [1, nstride, nstride, 1], padding = 'SAME', name = 'p')
    return p

def to_flat(x, output_dim, name):
    with tf.variable_scope(name):
        input_dim = x.get_shape()[-1]*x.get_shape()[-2]*x.get_shape()[-3]
        W = tf.get_variable(name = 'W', shape = [input_dim, output_dim],initializer = tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], initializer = tf.zeros_initializer())
        h_flat = tf.reshape(x, [-1, input_dim])
        h = tf.nn.relu(tf.matmul(h_flat, W)+b, name = 'h')
    return h
        
def linear_layer(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable('W', [x.get_shape()[-1], output_dim], initializer = tf.glorot_uniform_initializer())
        b = tf.get_variable('b', [output_dim], initializer = tf.zeros_initializer())
        h = tf.add(tf.matmul(x, W), b, name = 'h')
    return h

class simpleCNN(object):
    def __init__(self, sess):
        self.sess = sess
    
    def build_net(self, input_dim, nclass, width, height):
        self.input_dim = input_dim
        self.nclass = nclass
        self.X = tf.placeholder(dtype = tf.float32, shape = [None, self.input_dim], name = 'input_x')
        self.X_image = tf.reshape(self.X, [-1, width, height, 1], name = 'X_image')
        self.Y = tf.placeholder(dtype = tf.float32, shape = [None, self.nclass], name = 'input_y')
        self.learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
        self.beta1 = 0.5

        conv1 = conv2d(self.X_image, 5, 5, 32, 'conv1')
        pool1 = max_pool(conv1, 2, 2, 2, 'pool1')
        conv2 = conv2d(pool1, 5, 5, 64, 'conv2')
        pool2 = max_pool(conv2, 2, 2, 2, 'pool2')
        fc1 = to_flat(pool2, 1024, 'fully1')
        fc2 = linear_layer(fc1, 10, 'fully2')

        with tf.variable_scope('Softmax'):
            logits = tf.nn.softmax(fc2)

        with tf.variable_scope("Optimization"):
            self.loss = -tf.reduce_sum(self.Y*tf.log(logits))
            self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1).minimize(self.loss)        

        with tf.variable_scope("Accuracy"):
            self.correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

        with tf.variable_scope("Scalars"):            
            self.avg_loss = tf.placeholder(tf.float32)
            self.avg_acc = tf.placeholder(tf.float32)
            self.loss_scalar = tf.summary.scalar('loss', self.avg_loss)
            self.acc_scalar = tf.summary.scalar('acc', self.avg_acc)
            self.merged = tf.summary.merge_all()
        
    def train(self, mnist, learning_rate, total_epoch, batch_size, save_dir):
        self.sess.run(tf.global_variables_initializer())
        
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        writer = tf.summary.FileWriter(save_dir)
        writer.add_graph(self.sess.graph)
        
        total_step = int(len(mnist.train.images)//batch_size)
        u = learning_rate
        for epoch in range(total_epoch):
            avg_loss = 0
            avg_acc = 0
            
            for step in range(total_step):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, l, a = self.sess.run([self.optim, self.loss, self.accuracy], feed_dict = {self.X:batch_x, self.Y:batch_y, self.learning_rate:u})
                avg_loss += l/total_step
                avg_acc += a/total_step
            summary, epoch_loss, epoch_acc = self.sess.run([self.merged, self.loss_scalar, self.acc_scalar], feed_dict={self.avg_loss : avg_loss, self.avg_acc : avg_acc})
            writer.add_summary(summary, global_step = epoch)
            print("Epoch {:3d}, Loss : {:.6f}, Accuracy : {:.2%}".format(epoch, avg_loss, avg_acc))
            u = u*0.9
                                                  
mnist = input_data.read_data_sets("./data/mnist/", one_hot = True)
print("Data Shape : ",tf.convert_to_tensor(mnist.train.images).get_shape())

sess = tf.Session()
m = simpleCNN(sess)
m.build_net(784,10, 28, 28)
m.train(mnist, 0.001, 8, 200,"./Basic04_board")

'''
Epoch   0, Loss : 35.352553, Accuracy : 94.51%
Epoch   1, Loss : 8.523958, Accuracy : 98.72%
Epoch   2, Loss : 5.489989, Accuracy : 99.16%
Epoch   3, Loss : 3.592700, Accuracy : 99.44%
Epoch   4, Loss : 2.433586, Accuracy : 99.65%
Epoch   5, Loss : 1.725245, Accuracy : 99.73%
Epoch   6, Loss : 1.383204, Accuracy : 99.79%
Epoch   7, Loss : 0.943458, Accuracy : 99.84%
'''
