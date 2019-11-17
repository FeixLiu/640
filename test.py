import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_poo_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.device('/gpu:0'):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    xs = tf.placeholder(shape=[None, 784], dtype=tf.float32)
    ys = tf.placeholder(shape=[None, 10], dtype=tf.float32)
    keep_prob = tf.placeholder(dtype=tf.float32)

    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    w_conv1 = weight_variable(shape=[5, 5, 1, 32])
    b_conv1 = bias_variable(shape=[32])
    h_conv1 = tf.nn.relu(tf.add(conv2d(x_image, w_conv1), b_conv1))
    h_pool1 = max_poo_2x2(h_conv1)

    w_conv2 = weight_variable(shape=[5, 5, 32, 64])
    b_conv2 = bias_variable(shape=[64])
    h_conv2 = tf.nn.relu(tf.add(conv2d(h_pool1, w_conv2), b_conv2))
    h_pool2 = max_poo_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, shape=[-1, 7 * 7 * 64])
    w_fc1 = weight_variable(shape=[7 * 7 * 64, 1024])
    b_fc1 = bias_variable(shape=[1024])
    h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool2_flat, w_fc1), b_fc1))

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable(shape=[1024, 10])
    b_fc2 = bias_variable(shape=[10])
    prediction = tf.nn.softmax(tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2))

    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(ys * tf.log(prediction),
                       reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    for i in range(100000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        feed_dict = {
            xs: batch_xs,
            ys: batch_ys,
            keep_prob: 0.5
        }
        sess.run(train_step, feed_dict=feed_dict)
        if i % 50 == 0:
            print(sess.run(cross_entropy, feed_dict=feed_dict))
