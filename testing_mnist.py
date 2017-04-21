"""
training_tkwoo.py

Created on Fri Apr 21 2017
@author : tkwoo

RESNET for MNIST

"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import resnet_model
import sys
import random
import time
import cv2
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

img_size = 28
tf.set_random_seed(777)

mnist = input_data.read_data_sets("./data", one_hot=True)

saver_model_path = './model/mnist_weight/resnet-model-7'

# model_name = "resnet-model"
# log_path_tb = './logs/resnettest2/' + model_name

### hyperparameter for network
batch_size = 1
num_classes = 10

### tf input
with tf.name_scope('input'):
    learning_rate = tf.placeholder(tf.float32, [], name="learningrate")
    X = tf.placeholder(tf.float32, [None, 784], name="images_flat")
    images_tf = tf.reshape(X, [-1, 28, 28, 1], name="images")
    labels_tf = tf.placeholder(tf.int64, [None, 10], name="Labels")
    label_tf = tf.placeholder(tf.int64, [None], name="Labels2")

train_mode = tf.placeholder(tf.bool)

### resnet model
hps = resnet_model.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.00001,
                             lrn_rate=learning_rate,
                             num_residual_units=4,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='adam'
                             )
model = resnet_model.ResNet(hps, images_tf, labels_tf, 'eval')
model.build_graph()

saver = tf.train.Saver()

truth = tf.argmax(model.labels, axis=1)
predictions = tf.argmax(model.predictions, axis=1)
precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
# tf.summary.scalar('accuracy', precision)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

merged = tf.summary.merge_all()
# if tf.gfile.Exists(log_path_tb):
    # tf.gfile.DeleteRecursively(log_path_tb)
# writer = tf.summary.FileWriter(log_path_tb + "/test", sess.graph)

# tf.global_variables_initializer().run()
saver.restore( sess, saver_model_path )

with tf.device('/gpu:0'):
    start_time = time.time()
    test_precision_val, cost_val = sess.run(
                [precision, model.cost],
                feed_dict={X: mnist.test.images, labels_tf: mnist.test.labels, train_mode: False}
                )
    end_time = time.time() - start_time
    whole_time = end_time
    
    acc_all = test_precision_val
    print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print ('acc:'+str(acc_all), "cost:"+str(cost_val))
    print ('time : %.3f sec'%whole_time)
    print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    
