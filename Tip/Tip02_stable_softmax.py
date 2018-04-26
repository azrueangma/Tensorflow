def softmax(a):
    b = tf.reduce_max(a, axis=1, keepdims = True )
    c = tf.exp(tf.subtract(a,b))
    d = tf.reduce_sum(c, axis=1, keepdims = True)
    result = tf.divide(c,d)
    return result
