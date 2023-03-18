#!/usr/bin/env python
import cPickle, sys
import math, numpy, time, cv2, copy

from operator import itemgetter, attrgetter

import rospy, roslib
from nav_msgs.msg import Odometry
import tf as tf_ros

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

from os.path import join
import sys
import os 
import numpy as np


#from pyfann import libfann

import message_filters
#from message_filters import ApproximateTimeSynchronizer

from geometry_msgs.msg import PoseArray, Pose

from multiprocessing import Lock

import tensorflow as tf

def convert_image(data):
    try:
      cv_image = bridge.imgmsg_to_cv2(data, "bgr8")  
    except CvBridgeError, e:
      print e

    cv_image = numpy.asarray(cv_image)
    #cv2.imshow("Image window", cv_image)
    #cv2.waitKey(1)

    return cv_image
        
def scale(scale):

    maxX = 4.75


    maxY = 10.25
    
    scale[0] = (scale[0]) * maxX
    scale[1] = (scale[1]) * maxY

    
    scale[2] *= 2
    scale[2] -= 1
    scale[3] *= 2
    scale[3] -= 1
        
    return scale
def rescale(scale):    
    # min and max values are measured in gazebo
    minX = -3.5
    maxX = 4.75
    
    minY = -10.25
    maxY = 6.5
    
    # normalize the x 
    label[:, 0] /= maxX
    label[:, 0] = (label[:, 0]+1)/2
    
    #normalize the y
    label[:, 1] /= maxY
    label[:, 1] = (label[:, 1]+1)/2
    
    # normalize the sin and cos by adding 1 and divide by 2
    label[:, 2] += 1
    label[:, 2] /= 2
    label[:, 3] += 1
    label[:, 3] /= 2

    return scale

class Tester(object):
    def __init__(self):
        self.transformer = tf_ros.TransformListener()
        self.publisher = rospy.Publisher('estimated_pose', PoseArray, queue_size = 1)
        self.pose_publisher = rospy.Publisher('estimate_pose', Pose, queue_size = 1)
        
        self.location_array = PoseArray()
        self.location_array.header.frame_id="odom"
        self.lock = Lock()
        
        self.counter = 0
        
    def gmapping_image(self, input_image):
        global mean_diffx
        global mean_diffy
        global mean_total
        global index
        global average_image
        global counter

        
        
        # Converting Image to Grayscale, and normalizing
        input_image = convert_image(input_image)
           
        image_rgb = copy.deepcopy(input_image)

        cv2.imshow("cam", image_rgb)
        cv2.waitKey(1)
        
        #image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        #image_rgb = cv2.resize(image_rgb, (84, 84), interpolation = cv2.INTER_AREA)
        image_rgb = cv2.resize(image_rgb, (84, 84))
        image_rgb = numpy.asarray(image_rgb, dtype=numpy.float32)
        #image_rgb /= 127.5
        #image_rgb -= 1
	image_rgb /= 255
              
        
        
        #CNN HERE    
        output = L9.eval(session = sess, feed_dict = {x: [image_rgb], keep_prob: 1.0})
        #output = sess.run([L9], feed_dict={x: [image_rgb], keep_prob: 1.0})
        print output
        #print output

        x_loc,y_loc, sin_theta, cos_theta = output[0]
        
        label_names = ["x", "y", "sin", "cos"]
        #labels_norm = [2.375, 1.875, 1]
        labels_norm = [2.4, 1]
        labels_shift = [2.05, -1.55, 0, 0]

        #min_x = -0.25
        #max_x = 4.5
        #min_y = -3.5
        #max_y = 0.25
        #mean_x = (min_x+max_x)/2
        #mean_y = (min_y+max_y)/2

        x_loc = x_loc * 2.4 + 2.05
        y_loc = y_loc * 2.4 - 1.55
        
        theta = math.atan2(sin_theta, cos_theta)
        print x_loc, y_loc, math.degrees(theta)
        somePose = Pose()
        
        somePose.position.x = x_loc
        somePose.position.y = y_loc
        somePose.position.z = 1.0

        quaternion = tf_ros.transformations.quaternion_from_euler(0, 0, theta, 'sxyz')
        
        somePose.orientation.x = quaternion[0]
        somePose.orientation.y = quaternion[1]
        somePose.orientation.z = quaternion[2]
        somePose.orientation.w = quaternion[3]
        #self.lock.acquire()
        self.pose_publisher.publish(somePose)
        self.location_array.poses.append(somePose)
        #self.lock.release()
        
        self.update()
        

    def update(self):
        if self.counter > 3:
         #   self.lock.acquire()
         #   self.lock.release()
            
            self.counter = 0
            self.location_array.header.stamp = rospy.Time.now()
            self.publisher.publish(self.location_array)
            self.location_array.poses = []
        else:
            self.counter += 1
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME'):
    return tf.nn.conv2d(x, W, strides, padding='SAME')

def visualize_weight(weight):
    return tf.transpose (weight,[3, 0, 1, 2])
    

'''
PLEASE READ THIS:

Model: Make sure your model to the path is correct.
REQUIRED CNN LINES: For each model you need the correct CNN layers in the main functions
IMAGE: Make sure you are correctly dividing the image. It is either image = image / 127.5 - 1 or image /= 255.
IMAGE: The current dataset use BGR, so do not convert to RGB
RESCALING: THIS IS VERY IMPORTANT, make sure you have the right normalization values
'''
if __name__ == '__main__':

    test_step = 10
    checkpoint_step = 1

    inputWidth = 84
    inputHeight = 84
    inputChannels = 3

    nOutput = 4


    learning_rate = 0.00001
    training_epochs = 1000
    batchSize = 50

    directory = "regression/lr{}".format(learning_rate)

    label_names = ["x", "y", "sin", "cos"]
    #labels_norm = [5.125, 1] #not for kitchen? I don't now



    nKernelsL1 = 64
    kernelSizeL1 = 4

    nKernelsL2 = 64
    kernelSizeL2 = 4
    
    nKernelsL3 = 64
    kernelSizeL3 = 4

    nKernelsL4 = 64
    kernelSizeL4 = 4

    nKernelsL5 = 64
    kernelSizeL5 = 3

    FC1Size = 512

    x = tf.placeholder(tf.float32, shape=[None, inputHeight, inputWidth, inputChannels])
    y_ = tf.placeholder(tf.float32, shape=[None, nOutput])
    keep_prob = tf.placeholder(tf.float32)


    image = tf.reshape(x, [-1, inputWidth, inputHeight, inputChannels])
    tf.summary.image("images", [image[0]], max_outputs = 1)
    
    with tf.name_scope('conv1'):
        W1 = weight_variable([kernelSizeL1, kernelSizeL1, inputChannels, nKernelsL1])
        tf.summary.histogram("W1", W1)
        B1 = bias_variable([nKernelsL1])
        
        L1 = tf.nn.relu(conv2d(image, W1, [1,1,1,1]) + B1)

    with tf.name_scope('conv2a'):
        W2a = weight_variable([3, 3, nKernelsL1, nKernelsL2])
        tf.summary.histogram("w2a", W2a)
        B2a = bias_variable([nKernelsL2])
        
        L2a = tf.nn.relu(conv2d(L1, W2a, [1,2,2,1]) + B2a)

    with tf.name_scope('conv2b'):
        W2b = weight_variable([5, 5, nKernelsL1, nKernelsL2])
        tf.summary.histogram("w2b", W2b)
        B2b = bias_variable([nKernelsL2])
        
        L2b = tf.nn.relu(conv2d(L1, W2b, [1,2,2,1]) + B2b)

    L2 = tf.concat([L2a, L2b], 3)
#-----------------------------------------------------------------

    with tf.name_scope('conv3a'):
        W3a = weight_variable([3, 3, 2*nKernelsL2, nKernelsL2])
        tf.summary.histogram("w3a", W3a)
        B3a = bias_variable([nKernelsL2])
        
        L3a = tf.nn.relu(conv2d(L2, W3a, [1,1,1,1]) + B3a)

    with tf.name_scope('conv3b'):
        W3b = weight_variable([5, 5, 2*nKernelsL2, nKernelsL2])
        tf.summary.histogram("w3b", W3b)
        B3b = bias_variable([nKernelsL2])
        
        L3b = tf.nn.relu(conv2d(L2, W3b, [1,1,1,1]) + B3b)

    L3 = tf.concat([L3a, L3b], 3)

#----------------------------------------------------------------------------------

    with tf.name_scope('conv4a'):
        W4a = weight_variable([3, 3, 2*nKernelsL2, nKernelsL3])
        tf.summary.histogram("w4a", W4a)
        B4a = bias_variable([nKernelsL3])
        
        L4a = tf.nn.relu(conv2d(L3, W4a, [1,2,2,1]) + B4a)

    with tf.name_scope('conv4b'):
        W4b = weight_variable([5, 5, 2*nKernelsL2, nKernelsL3])
        tf.summary.histogram("w4b", W4b)
        B4b = bias_variable([nKernelsL2])
        
        L4b = tf.nn.relu(conv2d(L3, W4b, [1,2,2,1]) + B4b)

    L4 = tf.concat([L4a, L4b], 3)

#----------------------------------------------------------------------------------

    with tf.name_scope('conv5a'):
        W5a = weight_variable([3, 3, 2*nKernelsL3, nKernelsL3])
        tf.summary.histogram("w5a", W5a)
        B5a = bias_variable([nKernelsL3])
        
        L5a = tf.nn.relu(conv2d(L4, W5a, [1,2,2,1]) + B5a)

    with tf.name_scope('conv5b'):
        W5b = weight_variable([5, 5, 2*nKernelsL3, nKernelsL3])
        tf.summary.histogram("w5b", W5b)
        B5b = bias_variable([nKernelsL2])
        
        L5b = tf.nn.relu(conv2d(L4, W5b, [1,2,2,1]) + B5b)

    L5 = tf.concat([L5a, L5b], 3)

#-----------------------/home/borg/amir/models/kitchen/2/iteration1000-----------------------------------------------------------

    with tf.name_scope('conv6a'):
        W6a = weight_variable([3, 3, 2*nKernelsL3, nKernelsL4])
        tf.summary.histogram("w6a", W6a)
        B6a = bias_variable([nKernelsL4])
        
        L6a = tf.nn.relu(conv2d(L5, W6a, [1,1,1,1]) + B6a)

    with tf.name_scope('conv6b'):
        W6b = weight_variable([5, 5, 2*nKernelsL3, nKernelsL4])
        tf.summary.histogram("w6b", W6b)
        B6b = bias_variable([nKernelsL2])
        
        L6b = tf.nn.relu(conv2d(L5, W6b, [1,1,1,1]) + B6b)

    L6 = tf.concat([L6a, L6b], 3)

#----------------------------------------------------------------------------------

    with tf.name_scope('conv7a'):
        W7a = weight_variable([1, 1, 2*nKernelsL4, nKernelsL5])
        tf.summary.histogram("w7a", W7a)
        B7a = bias_variable([nKernelsL5])
        
        L7a = tf.nn.relu(conv2d(L6, W7a, [1,1,1,1]) + B7a)

    with tf.name_scope('conv7b'):
        W7b = weight_variable([3, 3, 2*nKernelsL4, nKernelsL5])
        tf.summary.histogram("w7b", W7b)
        B7b = bias_variable([nKernelsL2])
        
        L7b = tf.nn.relu(conv2d(L6, W7b, [1,1,1,1]) + B7b)

    L7 = tf.concat([L7a, L7b], 3)
    L7_out = tf.expand_dims(tf.transpose (L7, [0, 3, 1, 2]),4)[0]
    L7 = tf.nn.dropout(L7, keep_prob)


#----------------------------------------------------------------------------------
        
    L7 = tf.reshape(L7, [-1, 2*7744])
    
    with tf.name_scope('fc1'):
        W8 = weight_variable([2*7744, FC1Size])
        tf.summary.histogram("W8", W8)
        B8 = bias_variable([FC1Size])
        
        L8 = tf.nn.relu(tf.matmul(L7, W8) + B8)

    with tf.name_scope('fc2'):
        W9 = weight_variable([FC1Size, nOutput])
        tf.summary.histogram("w9", W9)
        B9 = bias_variable([nOutput])
        
        L9 = tf.matmul(L8, W9) + B9



    # Moving Average Variables
    mean_diffx = 0
    mean_diffy = 0
    mean_total = 0
    index = 0
    counter = 1
    average_image = None
    model = '/opt/enacer/sources/borg/latest_model/small_kitchen/model.ckpt'
    #config = tf.ConfigProto(
    #    device_count = {'GPU': 1}
    #)
    #sess = tf.Session(config=config)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, model)
    
    print "Model Loaded."
    
    shape = (84, 84)
    bridge = CvBridge()
    
    
    rospy.init_node('live_error_test', anonymous=True)
    rospy.loginfo("Node initialized")
    
    #Initializing the testing class
    test_hsv = Tester()
    
    ts = rospy.Subscriber("/sudo/bottom_webcam/image_raw", Image, test_hsv.gmapping_image)
    #ts = rospy.Subscriber("/front_xtion/rgb/image_raw", Image, test_hsv.gmapping_image)

    rospy.loginfo("Callbacks registered")   
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()
