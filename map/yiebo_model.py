import tensorflow as tf


'''
Inception network model
'''
def incep_block(x, filters, kernel1, kernel2, stride=1):

    with tf.variable_scope('layer_1'):
        x1 = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=kernel1, strides=stride, padding='SAME')
        x1 = tf.nn.relu(x1)

    with tf.variable_scope('layer_2'):
        x2 = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=kernel2, strides=stride, padding='SAME')
        x2 = tf.nn.relu(x2)
        
    x = tf.concat([x1, x2], 3)

    return x


def model(x, drop_rate):

    with tf.variable_scope('layer_0'):
        x = tf.layers.conv2d(inputs=x, filters=64,
                             kernel_size=4, strides=1, padding='SAME')
        x = tf.nn.relu(x)
       
    strides = [2, 1, 2, 2, 1]
    
    for i, stride in enumerate(strides, 1):
        with tf.variable_scope('layer_{}'.format(i)):
            x = incep_block(x, 64, 3, 5, stride)

    with tf.variable_scope('layer_6'):
        x = incep_block(x, 64, 1, 3)

    x = tf.nn.dropout(x, keep_prob=drop_rate)

    x = tf.layers.flatten(x)
    
    with tf.variable_scope('layer_7'):
        x = tf.layers.dense(x, 512)
        x = tf.nn.relu(x)
        
    with tf.variable_scope('layer_8'):
        x = tf.layers.dense(x, 4)
    return x



