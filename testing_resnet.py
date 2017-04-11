"""
training_tkwoo.py

Created on Thu Jun 19 2017
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

img_size = 28
tf.set_random_seed(777)

mnist = input_data.read_data_sets("./data", one_hot=True)

# print mnist.test.images.shape
# print mnist.test.images.reshape([-1,28,28,1]).shape
# exit()

saver_model_path = './model/mnist_small_non/resnet-model-7' #None #'./model_tf/resnet-model-1' 

model_name = "resnet-model"
log_path_tb = './logs/resnettest2/' + model_name

### hyperparameter for network
batch_size = 1

num_classes = 10

### tf input
with tf.name_scope('input'):
    learning_rate = tf.placeholder(tf.float32, [], name="learningrate")
    # images_tf = tf.placeholder(tf.float32, [None, img_size, img_size, 3], name="images")
    # labels_tf = tf.placeholder(tf.int64, [None], name="labels")
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
classmap = model.get_classmap(label_tf, model.last_unit, input_size=img_size)
saver = tf.train.Saver()

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
writer = tf.summary.FileWriter(log_path_tb + "/test", sess.graph)

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
    
    index_images = 3
    cam_image = mnist.test.images[index_images:index_images+1]

    output_vals, lu_val = sess.run(
            [model.predictions, model.last_unit],
            feed_dict={X: cam_image, train_mode: False}
            )
    label_predictions = output_vals.argmax(axis=1)

    classmap_vals = sess.run(classmap,
                feed_dict={
                    label_tf: label_predictions,
                    model.last_unit: lu_val
                })
    # print "DT:", label_predictions[0]
    # print "GT:", current_labels[0]
    current_images = cam_image.reshape([-1,28,28,1])
    classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_vals)
    for vis, ori,l_number, ErrVal in zip(classmap_vis, current_images, label_predictions, output_vals):
        print "Predict:",l_number
        print "softmax:",ErrVal[1]
        # b,g,r = cv2.split(ori)
        # ori = cv2.merge([r,g,b])
        ori = cv2.cvtColor(ori, cv2.COLOR_GRAY2RGB)
        plt.imshow( ori )
        fig = plt.imshow( vis, cmap=plt.cm.jet, 
                            alpha=0.4+0.1, 
                            interpolation='nearest' )
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        
    #     # if not os.path.isdir(output_path+fName_subjectNum):
    #     #     os.mkdir(output_path+fName_subjectNum)
    #     # # print Label_GT_mri_list[l_number]

    #     # p = "%.3f"%output_vals[0][1]
    #     # plt.savefig(output_path+fName_subjectNum+'/'
    #     #             # +Label_GT_cifar_list[l_number]+'_'+fName_input_image+'.png'
    #     #             +Label_GT_mri_list[l_number]+'_'+p+'_'+fName_input_image
    #     #             +'.png'
    #     #             , pad_inches=0., bbox_inches='tight', frameon=False, transparent=True)
        
        plt.show()
        
    
    

