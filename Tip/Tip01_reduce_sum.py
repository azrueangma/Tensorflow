#reduce_sum
'''
temp = self.Y_one_hot * tf.log(self.hypothesis)
v = tf.reshape(temp,[1,-1])
self.cost = -tf.reshape(tf.matmul(v, tf.ones_like(v), transpose_b=True),[],name = 'cost')  
'''

import tensorflow as tf
sess = tf.Session()

a = tf.constant([[1,2,3],[4,5,6]],dtype=tf.float32)
b = tf.reduce_sum(a)
print("tf.reduce_mean : ",sess.run(b))

c = tf.reshape(a,[1,-1])
d = tf.reshape(tf.matmul(c, tf.ones_like(c), transpose_b=True),[])
print("my reduce_sum : ", sess.run(d))
