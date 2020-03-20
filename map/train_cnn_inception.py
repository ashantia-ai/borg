#!/usr/bin/env python
import sys
import os 
import math
import numpy as np
import tensorflow as tf
from yiebo_model import *
import tqdm

def read_set(path, idx):
    images = (np.load("{}/images{}.npy".format(path, idx))/127.5)-1
    labels = np.load("{}/labels{}.npy".format(path, idx))
    idx_set = np.arange(labels.shape[0])
    np.random.shuffle(idx_set)
    images = np.take(images, idx_set, 0)
    labels = np.take(labels, idx_set, 0)
    return images, labels

'''
This files train the CNN based on the available numpy files. This file uses a 9 layer network, with Adam optimization and learening decay.
Make sure labals_norm is correct. Take care about the read_set function. it is divinding the image by 127.5. you need to make sure that you do the samee for using the network
'''

if __name__ == '__main__' :
    train_size = len(os.listdir("data/train"))//2
    test_size = len(os.listdir("data/test"))//2
    print(train_size)
    print(test_size)
    test_step = 2
    checkpoint_step = 1

    learning_rate = 0.0001
    training_epochs = 1000
    batchSize = 32

    directory = "regression/lr{}".format(learning_rate)

    label_names = ["x", "y", "sin", "cos"]
    labels_norm = [2.7435, 1]


    x = tf.placeholder(tf.float32, shape=[None, 84, 84, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 4])
    drop_rate = tf.placeholder(tf.float32)

    with tf.variable_scope('model'):
        y = model(x, drop_rate)


    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.pow(y - y_, 2))
        
        tf.train.create_global_step()
        global_step = tf.train.get_global_step()
        learning_rate_ = tf.train.exponential_decay(learning_rate, global_step,
                                                    decay_steps=500, decay_rate=0.95)
        train_step = tf.train.AdamOptimizer(learning_rate_, ).minimize(cost, global_step=global_step)


    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        saver = tf.train.Saver(max_to_keep=100)


        if not os.path.exists(directory):
            os.makedirs(directory)
        run_nr=len(os.listdir(directory))
        directory = "{}/{}".format(directory, run_nr)

        directory_model = "{}/model".format(directory)
        if not os.path.exists(directory_model):
            os.makedirs(directory_model)

        writer = tf.summary.FileWriter(directory, sess.graph) # for 0.8
        print(directory)


        #Step 12 train the  model
        for i in tqdm.trange(training_epochs+1):
            train_cost = 0
            batch_idx = 0
            error = np.zeros(2)
            idx_set = np.arange(train_size)
            np.random.shuffle(idx_set)
            
            for j in tqdm.tqdm(idx_set[:2]):
                images, labels = read_set("data/train", j)
                for idx in tqdm.trange(0,len(labels),batchSize):
                    _, c, prediction, current_lr = sess.run([train_step, cost, y, learning_rate_], feed_dict={x: images[idx:idx+batchSize], y_: labels[idx:idx+batchSize], drop_rate: 0.5})
                    train_cost += c
                    a = np.empty((len(labels[idx:idx+batchSize]), 3))

                    a[:,[0,1]]=prediction[:,[0,1]]
                    a[:,2]=np.arctan2(prediction[:,2], prediction[:,3])

                    b=labels[idx:idx+batchSize,[0,1,2]]
                    b[:,2]=np.arctan2(labels[idx:idx+batchSize][:,2], labels[idx:idx+batchSize][:,3])
                    sub_error = np.mean(np.absolute(a-b),0)
                    error += [math.hypot(sub_error[0],sub_error[1]), sub_error[2]]
                    batch_idx += 1
            error= error/batch_idx
            error*=labels_norm
            error[1] = math.degrees(error[1])
            train_cost /= batch_idx

            value_train = tf.Summary.Value(tag="cost/train", simple_value=train_cost)
            summary_cost_train = tf.Summary(value=[value_train])
            writer.add_summary(summary_cost_train, i)
            

            summary_error_meter = tf.Summary.Value(tag="meter/train", simple_value=error[0])
            summary_error_meter = tf.Summary(value=[summary_error_meter])
            writer.add_summary(summary_error_meter, i)

            summary_error_angle = tf.Summary.Value(tag="angle/train", simple_value=error[1])
            summary_error_angle = tf.Summary(value=[summary_error_angle])
            writer.add_summary(summary_error_angle, i)
            
            summary_lr = tf.Summary(value=[tf.Summary.Value(tag="util/train", simple_value=current_lr)])
            writer.add_summary(summary_lr, i)
            



            if i % checkpoint_step == 0:
                if not os.path.exists('{}/iteration{}'.format(directory_model, i)):
                    os.makedirs('{}/iteration{}'.format(directory_model,i))
                saver.save(sess, '{}/iteration{}/model.ckpt'.format(directory_model, i))
            if i % test_step == 0:
                test_cost=0
                batch_idx=0
                error = np.zeros(2)
                idx_set = np.arange(test_size)
                np.random.shuffle(idx_set)
                for idx in tqdm.tqdm(idx_set[:1]):
                    images, labels = read_set("data/test", idx)
                    for jdx in tqdm.trange(0,len(labels),batchSize):
                        c, prediction = sess.run([cost, y], feed_dict={x: images[jdx:jdx+batchSize], y_: labels[jdx:jdx+batchSize], drop_rate: 1.0})
                        test_cost+=c
                        a = np.empty((len(labels[jdx:jdx+batchSize]), 3))

                        a[:,[0,1]]=prediction[:,[0,1]]
                        a[:,2]=np.arctan2(prediction[:,2], prediction[:,3])

                        b=labels[jdx:jdx+batchSize,[0,1,2]]
                        b[:,2]=np.arctan2(labels[jdx:jdx+batchSize][:,2], labels[jdx:jdx+batchSize][:,3])
                        sub_error = np.mean(np.absolute(a-b),0)
                        error += [math.hypot(sub_error[0],sub_error[1]), sub_error[2]]
                        batch_idx += 1
                error= error/batch_idx
                error*=labels_norm
                error[1] = math.degrees(error[1])
                test_cost /= batch_idx

                value_test = tf.Summary.Value(tag="cost/test", simple_value=test_cost)
                summary_cost_test = tf.Summary(value=[value_test])
                writer.add_summary(summary_cost_test, i)
            
                summary_error_meter = tf.Summary.Value(tag="meter/test", simple_value=error[0])
                summary_error_meter = tf.Summary(value=[summary_error_meter])
                writer.add_summary(summary_error_meter, i)

                summary_error_angle = tf.Summary.Value(tag="angle/test", simple_value=error[1])
                summary_error_angle = tf.Summary(value=[summary_error_angle])
                writer.add_summary(summary_error_angle, i)

                

