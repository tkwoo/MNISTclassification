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
from tensorflow.examples.tutorials.mnist import input_data

img_size = 28
tf.set_random_seed(777)
### input source
### dataset_path must have directory format like '001.labelname'
### any file name is ok
mnist = input_data.read_data_sets("./data", one_hot=True)

# dataset_path = '../../01.dataset_local/ANTsBMLabel_train' #cifar_b_all' #bm_gtcc1'
pretrained_model_path = None#'./model/mnist_small1/resnet-model-2'
# FileType = '.png' #'png'

### output path
model_name = "resnet-model"
saver_model_path = './model/mnist_small_non/' + model_name
log_path_tb = './logs/mnist_small_non/' + model_name
val_log_txt_path = './log.tmp.txt'

### hyperparameter for network
batch_size = 300
init_learning_rate = 0.003   #AdamOptimizer is used. GAP0.00007
# 10 epoch -> 0.345
n_epochs = 30 # when learning rate decay is 0.9, epoch 50 -> init learning rate/200
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
    # images_tf = tf.placeholder(tf.float32, [None, img_size, img_size, 3], name="images")
    # labels_tf = tf.placeholder(tf.int64, [None], name="labels")
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

param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

truth = tf.argmax(model.labels, axis=1)
predictions = tf.argmax(model.predictions, axis=1)
precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
tf.summary.scalar('accuracy', precision)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
# config.log_device_placement = True
sess = tf.InteractiveSession(config=config)

merged = tf.summary.merge_all()
if tf.gfile.Exists(log_path_tb):
    tf.gfile.DeleteRecursively(log_path_tb)
writer = tf.summary.FileWriter(log_path_tb, sess.graph)
# writer_val = tf.summary.FileWriter(log_path_tb + '/val')

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
            # sess.run(vgg.update_ema, feed_dict={learning_rate: init_learning_rate,
            #                     images_tf: current_images, 
            #                     labels_tf: current_labels, 
            #                     train_mode: True})
            
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
        
        # ### validation
        # n_correct = 0
        # n_data = 0
        
        # test_precision_val = sess.run(
        #         precision,
        #         feed_dict={X: mnist.test.images, labels_tf: mnist.test.labels, train_mode: False}
        #         )
        
        # acc_all = test_precision_val
        # #print (label_predictions)
        
        # # n_correct += acc
        # # n_data += len(current_data)
        
        # # acc_all = n_correct / float(n_data)
        # f_log.write('iter:'+str(iterations-1)+'\tepoch:'+str(epoch)+'\tacc:'+str(acc_all)+'\n')
        # print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print ('epoch:'+str(epoch)+'\tacc:'+str(acc_all))
        # print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        # save_path = saver.save(sess, saver_model_path+str(epoch)+'.ckpt')
        save_path = saver.save(sess, saver_model_path, global_step=epoch)
        print "file saved: ", save_path

        init_learning_rate *= lr_decay
        # sess.run(model.global_step, feed_dict={model.lrn_rate: init_learning_rate})

    writer.close()
