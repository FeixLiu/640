import csv

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import cv2
import os
import numpy as np
import math


def weight_variable( shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable( shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d( x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_poo_2x2( x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.device('/gpu:0'):
    xs = tf.placeholder(shape=[None, 64, 64, 1], dtype=tf.float32)
    ys = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    keep_prob = tf.placeholder(dtype=tf.float32)

    x_image = tf.reshape(xs, [-1, 64, 64, 1])

    w_conv1 = weight_variable(shape=[5, 5, 1, 32])
    b_conv1 = bias_variable(shape=[32])
    h_conv1 = tf.nn.relu(tf.add(conv2d(x_image, w_conv1), b_conv1))
    h_pool1 = max_poo_2x2(h_conv1)

    w_conv2 = weight_variable(shape=[5, 5, 32, 64])
    b_conv2 = bias_variable(shape=[64])
    h_conv2 = tf.nn.relu(tf.add(conv2d(h_pool1, w_conv2), b_conv2))
    h_pool2 = max_poo_2x2(h_conv2)

    image_sum = tf.reduce_mean(h_pool2, axis=0, keepdims=True)

    h_pool2_flat = tf.reshape(image_sum, shape=[-1, 16 * 16 * 64])
    w_fc1 = weight_variable(shape=[16 * 16 * 64, 1024])
    b_fc1 = bias_variable(shape=[1024])
    h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool2_flat, w_fc1), b_fc1))

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable(shape=[1024, 3])
    b_fc2 = bias_variable(shape=[3])
    prediction = tf.nn.softmax(tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2))
    a = tf.arg_max(prediction, dimension=1)

    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(ys * tf.log(prediction),
                       reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    
    labels = {}
    csvFile = open("./640/Labels.csv", "r")
    reader = csv.reader(csvFile)
    for item in reader:
        if reader.line_num == 1:
                    continue
        labels[item[0]] = item[1]
    csvFile.close()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        path = './test/'
        transfer = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
        for j in range(10):
            for im_name in os.listdir(path):
                label = labels[im_name + '.mp4']
                temp = [0 for _ in range(3)]
                temp[transfer[label]] = 1
                temp = np.array(temp)
                temp = temp[np.newaxis, :]
                test_path = os.path.join(path, im_name)
                video = []
                i = 0
                for img in os.listdir(test_path):
                    img_path = os.path.join(test_path, img)
                    im = cv2.imread(img_path)
                    im = cv2.resize(im, (64, 64))
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                    im = np.array(im)
                    im = im / 256
                    im = im[:, :, np.newaxis]
                    video.append(im)
                    i += 1
                    if i % 100 == 0:
                        break
                video = np.array(video)
                feed_dict = {
                    xs: video,
                    ys: temp,
                    keep_prob: 0.5
                }
                if j % 5 == 0:
                    print(sess.run(cross_entropy,feed_dict))
                    saver.save(sess, './checkpoint_dir/MyModel', global_step=j)



    # load model
    with tf.Session(config=config) as sess_load:
        # First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-10.meta')
        # saver.restore(sess_load, tf.train.latest_checkpoint('./'))
    
        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data
    
        graph = tf.get_default_graph()
        path2='./validate/'
        transfer = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
        for im_name in os.listdir(path2):
            label = labels[im_name + '.mp4']
            temp = [0 for _ in range(3)]
            temp[transfer[label]] = 1
            temp = np.array(temp)
            temp = temp[np.newaxis, :]
            path2_path = os.path.join(path2, im_name)
            video = []
            i = 0
            for img in os.listdir(path2_path):
                img_path = os.path.join(path2_path, img)
                im = cv2.imread(img_path)
                im = cv2.resize(im, (64, 64))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                im = np.array(im)
                im = im / 256
                im = im[:, :, np.newaxis]
                video.append(im)
                i += 1
                if i % 100 == 0:
                    break
            video = np.array(video)
            feed_dict = {
                xs: video,
                keep_prob: 0.5
            }
    
    
            # Now, access the op that you want to run.
            #op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
    
            print(sess_load.run(prediction, feed_dict))
    # tf.train.import_meta_graph('my_test_model-1000.meta')
