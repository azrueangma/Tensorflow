import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import shutil
import os

tf.set_random_seed(2)
npoints = 10000
save_dir = "./Basic01_board"

for i in range(npoints):
    np.random.seed(i)
    x = np.random.normal(0.0, 0.5)
    noise = np.random.normal(0.0, 0.05)
    y = x*0.2+0.5+noise
    tmp = np.expand_dims(np.array([x, y]), axis=0)
    if i == 0:
        vectors = tmp
    else:
        vectors = np.append(vectors, tmp,axis=0)
        
trainX = np.expand_dims(vectors[:,0],axis=1)
trainY = np.expand_dims(vectors[:,1],axis=1)

with tf.name_scope('Input'):
    X = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Y')

with tf.name_scope('Layer'):
    W = tf.Variable(tf.truncated_normal(shape = [1]), name = 'W')
    b = tf.Variable(tf.zeros([1]),name = 'b')
    h = W*X+b

with tf.name_scope('Optimizer'):
    loss = tf.reduce_mean(tf.square(Y-h))
    train = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)

avg_loss = tf.placeholder(tf.float32)
loss_scalar = tf.summary.scalar('loss', avg_loss)
merged = tf.summary.merge_all()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
writer = tf.summary.FileWriter(save_dir)
writer.add_graph(sess.graph)

miniBatchSize = 100
total_step = int(len(trainX)/miniBatchSize)
total_epoch = 80

plt.figure(num=None, figsize=(8, 14), dpi=60, facecolor='w', edgecolor='k')

cnt = 421
for epoch in range(total_epoch):
    np.random.seed(epoch)
    mask = np.random.permutation(len(trainX))
    step_loss = 0 
    for step in range(total_step):
        s = step*miniBatchSize
        t = (step+1)*miniBatchSize
        l, _ = sess.run([loss, train], feed_dict = {X:trainX[s:t,:], Y:trainY[s:t,:]})
        step_loss+=l/total_step
    summary, epoch_loss = sess.run([merged, loss_scalar], feed_dict = {avg_loss:step_loss})
    writer.add_summary(summary, global_step = epoch)
    if epoch%10==0:
        plt.subplot(cnt)
        plt.scatter(vectors[:,0], vectors[:,1],marker = '.')
        plt.plot(vectors[:,0], sess.run(W)*vectors[:,0]+sess.run(b),'r')
        plt.title('Epoch {}'.format(epoch))
        plt.grid()
        plt.xlim(-2,2)
        plt.ylim(0,1)
        cnt+=1
        print("Epoch {} Loss : {:.6f}".format(epoch, step_loss))
        
plt.suptitle('LinearRegression', fontsize=20)
plt.show()

