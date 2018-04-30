from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import shutil
import os

TOWER_NAME = 'Tower'
NUM_GPUS = 2
INPUT_DIM = 784
NCLASS = 10
LEARNING_RATE = 1E-2
TOTAL_EPOCH = 100000
BATCH_SIZE = 256
SAVE_DIR = './mnist_board'

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var

def linear_layer(x, input_dim, output_dim, name):
    with tf.variable_scope(name):
        W = _variable_on_cpu('W', [input_dim, output_dim], initializer = tf.glorot_uniform_initializer())
        b = _variable_on_cpu('b', [output_dim], initializer = tf.zeros_initializer())
        h = tf.add(tf.matmul(x, W), b, name = 'h')
    return h

def relu_layer(x, input_dim, output_dim, name):
    with tf.variable_scope(name):
        W = _variable_on_cpu('W', [input_dim, output_dim], initializer = tf.glorot_uniform_initializer())
        b = _variable_on_cpu('b', [output_dim], initializer = tf.zeros_initializer())
        h = tf.nn.relu(tf.add(tf.matmul(x, W), b), name = 'h')
    return h

def inference(images):
    h1 = relu_layer(images, 784, 1000, 'hidden1')
    h2 = relu_layer(h1, 1000, 500, 'hidden2')
    h3 = relu_layer(h2, 500, 200, 'hidden3')
    logits = linear_layer(h3, 200, 10, 'output')
    return logits

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def tower_loss(scope, images, labels):
    logits = inference(images)
    _ = loss(logits, labels)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def get_acc(images, labels):
    logits = inference(images)
    correct_prediction = tf.equal(tf.argmax(logits,1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name = 'accuracy')
    return accuracy

def train():
    mnist = input_data.read_data_sets("./data/mnist/", one_hot = False)
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)
        X = tf.placeholder(tf.float32, [None, 784])
        Y = tf.placeholder(tf.int64, [None])
        opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        tower_grads = []
        acc = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(NUM_GPUS):
                s = int(i*BATCH_SIZE/2)
                t = int((i+1)*BATCH_SIZE/2)
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                        image_batch = X[s:t];  label_batch = Y[s:t]
                        loss= tower_loss(scope, image_batch, label_batch)
                        tf.get_variable_scope().reuse_variables()
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)
                        a = get_acc(image_batch, label_batch)
                        acc.append(a)
        grads = average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        train_op = apply_gradient_op
        total_acc = tf.cast(tf.add_n(acc), tf.float32)
        init = tf.global_variables_initializer()
        config=tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess = tf.Session(config = config)
        sess.run(init)  
        tf.train.start_queue_runners(sess=sess)
        
        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)
        writer = tf.summary.FileWriter(SAVE_DIR)
        writer.add_graph(sess.graph)
        
        for epoch in range(TOTAL_EPOCH):
            batchX, batchY = mnist.train.next_batch(BATCH_SIZE)
            if epoch%1000==0:
                 _, loss_value, accu = sess.run([train_op, loss, total_acc], feed_dict = {X:batchX, Y:batchY})
                 print("Epoch : {:6d} Loss : {:.6f} Accuracy : {:.2%}".format(epoch,loss_value, accu/NUM_GPUS))

if __name__ == '__main__':
  train()

