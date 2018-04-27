from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import shutil
import os

def svmTarget(y_data, nclass):
    tmp = y_data*2
    tmp = tmp-1
    return tmp

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
        
def linear_layer(x, output_dim, name, with_w=False):
    with tf.variable_scope(name):
        W = tf.get_variable('W', [x.get_shape()[-1], output_dim], initializer = tf.glorot_uniform_initializer())
        b = tf.get_variable('b', [output_dim], initializer = tf.zeros_initializer())
        h = tf.add(tf.matmul(x, W), b, name = 'h')
    if with_w:
        return h, W
    else:
        return h

def drop_layer(x, keep_prob, name):
    with tf.variable_scope(name):
        h = tf.nn.dropout(x, keep_prob = keep_prob, name = 'h')
        return h
    
class SVM(object):
    def __init__(self, sess, name):
        self.name = name
        self.sess = sess
        
    def build_net(self, input_dim, nclass, width, height):
        with tf.variable_scope(self.name):
            with tf.variable_scope('Input'):
                self.input_dim = input_dim
                self.nclass = nclass
                self.X = tf.placeholder(dtype = tf.float32, shape = [None, self.input_dim], name = 'input_x')
                self.X_image = tf.reshape(self.X, [-1, width, height, 1], name = 'X_image')
                self.Y = tf.placeholder(dtype = tf.float32, shape = [None, self.nclass], name = 'input_y')
                self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
                self.learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')

            conv1 = conv2d(self.X_image, 5, 5, 32, 'conv1')
            pool1 = max_pool(conv1, 2, 2, 2, 'pool1')
            drop1 = drop_layer(pool1, self.keep_prob, 'drop1')
            conv2 = conv2d(drop1, 5, 5, 64, 'conv2')
            pool2 = max_pool(conv2, 2, 2, 2, 'pool2')
            fc1 = to_flat(pool2, 1024, 'fully1')
            model_output, W = linear_layer(fc1, 10, 'fully2', with_w = True)

            with tf.variable_scope("SVMLoss"):
                l2_norm = tf.reduce_sum(tf.square(W),axis=0)
                C = tf.constant([10.0])
                temp =  tf.subtract(1., tf.multiply(model_output, self.Y))
                hinge_loss = tf.reduce_sum(tf.square(tf.maximum(tf.zeros_like(temp),temp)),axis=0)
                self.loss = tf.reduce_sum(tf.add(tf.multiply(C,hinge_loss),l2_norm))
                
            with tf.variable_scope("Optimization"):
                self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
                
            with tf.variable_scope("Accuracy"):
                self.correct_prediction = tf.equal(tf.argmax(model_output,1), tf.argmax(self.Y, 1))
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
                _, l, a = self.sess.run([self.optim, self.loss, self.accuracy], feed_dict = {self.X:batch_x, self.Y:svmTarget(batch_y,10), self.learning_rate:u, self.keep_prob:0.7})
                avg_loss += l/total_step
                avg_acc += a/total_step
            summary, epoch_loss, epoch_acc = self.sess.run([self.merged, self.loss_scalar, self.acc_scalar], feed_dict={self.avg_loss : avg_loss, self.avg_acc : avg_acc})
            writer.add_summary(summary, global_step = epoch)
            print("Epoch {:3d}, Loss : {:.6f}, Accuracy : {:.2%}".format(epoch, avg_loss, avg_acc))
            u = u*0.9
                                                  
mnist = input_data.read_data_sets("./data/mnist/", one_hot = True)
print("Data Shape : ",tf.convert_to_tensor(mnist.train.images).get_shape())

sess = tf.Session()
m = SVM(sess, 'SVMModel')
m.build_net(784,10, 28, 28)
m.train(mnist, 0.001, 8, 200,"./Basic07_board")
'''
Epoch   0, Loss : 1173.785230, Accuracy : 92.75%
Epoch   1, Loss : 252.551959, Accuracy : 98.69%
Epoch   2, Loss : 167.892871, Accuracy : 99.14%
Epoch   3, Loss : 135.826198, Accuracy : 99.34%
Epoch   4, Loss : 110.266103, Accuracy : 99.47%
Epoch   5, Loss : 90.172499, Accuracy : 99.57%
Epoch   6, Loss : 75.350814, Accuracy : 99.69%
Epoch   7, Loss : 66.794713, Accuracy : 99.71%
'''
