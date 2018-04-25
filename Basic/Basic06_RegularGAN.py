from load_data import *
import tensorflow as tf
import numpy as np
import shutil

def maxout(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keepdims=False)
    return outputs
#from https://github.com/philipperemy/tensorflow-maxout/blob/master/maxout.py

def conv2d(x, kernel_width, kernel_height, kernel_channel, stride_width, stride_height, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name = 'W', shape = [kernel_width,  kernel_height, x.get_shape()[-1], kernel_channel], initializer = tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [kernel_channel], initializer = tf.zeros_initializer())
        h = tf.nn.conv2d(x, W, strides=[1,stride_width, stride_height,1], padding = 'SAME', name = 'h')
        conv = tf.reshape(tf.nn.bias_add(h, b), h.get_shape())
    return conv

def deconv2d(x, output_shape, kernel_width, kernel_height, stride_width, stride_height, name):
    with tf.variable_scope(name):
        W = tf.get_variable('W', [kernel_height, kernel_width, output_shape[-1], x.get_shape()[-1]], initializer = tf.glorot_uniform_initializer())
        b = tf.get_variable('b', [output_shape[-1]], initializer = tf.zeros_initializer())
        try:
            h = tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, stride_height, stride_width, 1], name = 'h')
        except AttributeError:
            h = tf.nn.deconv2d(x, w, output_shape=output_shape, strides=[1, stride_height, stride_width, 1], name = 'h')
        deconv = tf.reshape(tf.nn.bias_add(h, b), h.get_shape())
    return deconv

def linear_layer(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable('W', [x.get_shape()[-1], output_dim], initializer = tf.glorot_uniform_initializer())
        b = tf.get_variable('b', [output_dim], initializer = tf.zeros_initializer())
        h = tf.add(tf.matmul(x, W), b, name = 'h')
        return h

class GAN(object):
    tf.set_random_seed(0)
    def __init__(self, sess):
        self.sess = sess
        self.data_X, self.data_y = load_mnist()
        
    def discriminator(self, x, d_name, reuse=False):
        with tf.variable_scope(d_name, reuse=reuse):
            bs = int(x.get_shape()[0])
            net = maxout(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'),4)
            net = maxout(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'),4)
            net = tf.reshape(net, [bs, -1])
            net = maxout(linear_layer(net, 1024, name='d_fc3'),1024)
            out_logit = linear_layer(net, 1, name='d_fc4')
            out = tf.nn.sigmoid(out_logit)
            return out_logit

    def generator(self, z, g_name, reuse=False):
        with tf.variable_scope(g_name, reuse=reuse):
            bs = int(z.get_shape()[0])
            net = tf.nn.leaky_relu(linear_layer(z, 1024, name='g_fc1'))
            net = tf.nn.leaky_relu(linear_layer(net, 128 * 7 * 7, name='g_fc2'))
            net = tf.reshape(net, [bs, 7, 7, 128])
            net = tf.nn.leaky_relu(deconv2d(net, [bs, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'))
            out = tf.nn.sigmoid(deconv2d(net, [bs, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))
            return out

    def build_model(self, batch_size, noise_dim):
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.input_height = 28
        self.input_width = 28
        self.channel_dim = 1
          
        image_dims = [self.input_height, self.input_width, self.channel_dim]
        with tf.variable_scope("Input"):
            self.X = tf.placeholder(tf.float32, [batch_size] + image_dims, name='X')
            self.Z = tf.placeholder(tf.float32, [batch_size, self.noise_dim], name='z')
            self.learning_rate = tf.placeholder(tf.float32)
        
        D_real_logits= self.discriminator(self.X, 'Discriminator', reuse=False)
        G = self.generator(self.Z, 'Generator', reuse=False)
        D_fake_logits = self.discriminator(G, 'Discriminator', reuse=True)
        
        with tf.variable_scope("Loss"):
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
            self.d_loss = d_loss_real + d_loss_fake
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

        with tf.variable_scope("Optimization"):
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'd_' in var.name]
            g_vars = [var for var in t_vars if 'g_' in var.name]
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5).minimize(self.g_loss, var_list=g_vars)

    def train(self, total_epoch, learning_rate, save_dir):
        tf.global_variables_initializer().run()
        
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        writer = tf.summary.FileWriter(save_dir)
        writer.add_graph(self.sess.graph)
        
        total_step = int(len(self.data_X) // self.batch_size)
        u = learning_rate; seed = 0;
        for epoch in range(total_epoch):
            for step in range(total_step):
                batch_images = self.data_X[step*self.batch_size:(step+1)*self.batch_size]
                np.random.seed(seed); seed+=1;
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.noise_dim]).astype(np.float32)
                _,  d_loss = self.sess.run([self.d_optim, self.d_loss], feed_dict={self.X: batch_images, self.Z: batch_z, self.learning_rate:u})
                _,  g_loss = self.sess.run([self.g_optim, self.g_loss], feed_dict={self.Z: batch_z, self.learning_rate:u})
                if step%10==0:
                    print("Epoch: {:2d} {:4d}/{:4d}, d_loss: {:.8f}, g_loss: {:.8f}".format(epoch, step, total_step, d_loss, g_loss))
#refrence : https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/GAN.py
                    
with tf.Session() as sess:
    total_epoch = 5
    batch_size = 64
    z_dim = 1000
    gan = GAN(sess)
    gan.build_model(batch_size, z_dim)
    gan.train(total_epoch, 0.0001, "./Basic06_board")


