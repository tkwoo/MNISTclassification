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
from tensorflow.examples.tutorials.mnist import input_data

img_size = 28
tf.set_random_seed(777)
### input source
mnist = input_data.read_data_sets("./data", one_hot=True)
pretrained_model_path = None

### output path
model_name = "resnet-model"
saver_model_path = './model/mnist_weight/' + model_name
log_path_tb = './logs/mnist_weight/' + model_name
val_log_txt_path = './log.tmp.txt'

### hyperparameter for network
batch_size = 200
init_learning_rate = 0.003  
n_epochs = 10 
weight_decay_rate = 0.0005
n_valsetsize_per_class = 10
num_classes = 10
lr_decay = 0.5
init_optimizers = 'adam'

### log file open
f_log = open(val_log_txt_path, 'w')

### tf input
with tf.name_scope('input'):
    learning_rate = tf.placeholder(tf.float32, [], name="learningrate")
    X = tf.placeholder(tf.float32, [None, 784], name="images_flat")
    images_tf = tf.reshape(X, [-1, 28, 28, 1], name="images")
    labels_tf = tf.placeholder(tf.float32, [None, 10], name="Labels")

tf.summary.image('input', images_tf, 10)

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
                             optimizer=init_optimizers)
model = resnet_model.ResNet(hps, images_tf, labels_tf, 'train')
model.build_graph()
saver = tf.train.Saver(max_to_keep=50)

truth = tf.argmax(model.labels, axis=1)
predictions = tf.argmax(model.predictions, axis=1)
precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
tf.summary.scalar('accuracy', precision)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

merged = tf.summary.merge_all()
if tf.gfile.Exists(log_path_tb):
    tf.gfile.DeleteRecursively(log_path_tb)
writer = tf.summary.FileWriter(log_path_tb, sess.graph)

tf.global_variables_initializer().run()

if(pretrained_model_path):
    saver.restore(sess, pretrained_model_path)

with tf.device('/gpu:0'):

    iterations = 0
    loss_list = []    

    for epoch in range(n_epochs):
        
        total_batch = int(mnist.train.num_examples / batch_size)
        ### training
        
        for i in range(total_batch):
            ### actual training step
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, loss_val, output_val, precision_val = sess.run([model.train_op, model.cost, model.predictions,precision], feed_dict={
                                learning_rate: init_learning_rate,
                                X: batch_x, 
                                labels_tf: batch_y, 
                                train_mode: True
                                })
            loss_list.append(loss_val)
            
            if iterations % 5 == 0:
                print ("======================================")
                print ("Epoch", epoch, "Iteration", iterations)
                print ("Processed", i, '/', total_batch)

                acc = precision_val
                
                print ("Accuracy:", acc)
                print ("Training Loss:", np.mean(loss_list))
                
                loss_list = []
            if iterations % 20 == 0:
                summary = sess.run(merged, feed_dict = {learning_rate: init_learning_rate,
                                X: batch_x, 
                                labels_tf: batch_y, 
                                train_mode: True})
                writer.add_summary(summary, iterations)
                writer.flush()

            iterations += 1

        save_path = saver.save(sess, saver_model_path, global_step=epoch)
        print "file saved: ", save_path

        init_learning_rate *= lr_decay
        # sess.run(model.global_step, feed_dict={model.lrn_rate: init_learning_rate})

    writer.close()
