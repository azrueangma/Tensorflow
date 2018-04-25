from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import shutil
import os

def linear_layer(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable('W', [x.get_shape()[-1], output_dim], initializer = tf.glorot_uniform_initializer())
        b = tf.get_variable('b', [output_dim], initializer = tf.zeros_initializer())
    return tf.add(tf.matmul(x, W), b)

def relu_layer(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable('W', [x.get_shape()[-1], output_dim], initializer = tf.glorot_uniform_initializer())
        b = tf.get_variable('b', [output_dim], initializer = tf.zeros_initializer())
    return tf.nn.relu(tf.add(tf.matmul(x, W), b))
    
class simpleDNN(object):
    tf.set_random_seed(0)
    def __init__(self, sess, input_dim, nclass):
        self.sess = sess
        self.input_dim = input_dim
        self.nclass = nclass
        
    def build_net(self):
        self.X = tf.placeholder(dtype = tf.float32, shape = [None, self.input_dim], name = "input_X")
        self.Y = tf.placeholder(dtype = tf.float32, shape = [None, self.nclass], name = "input_Y")
        self.learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
        self.beta1 = 0.5
        
        h1 = relu_layer(self.X, 1000, 'hidden1')
        h2 = relu_layer(h1, 500, 'hidden2')
        h3 = relu_layer(h2, 200, 'hidden3')
        h4 = linear_layer(h3, 10, 'output')
        logits = tf.nn.softmax(h4)
        self.loss = -tf.reduce_sum(self.Y*tf.log(logits))
        self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1).minimize(self.loss)        
    
        self.correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
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
m = simpleDNN(sess, 784,10)
m.build_net()
m.train(mnist, 0.001, 8, 200,"./Basic03_board")
