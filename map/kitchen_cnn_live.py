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
from yiebo_model import model

def convert_image(data):
    try:
      cv_image = bridge.imgmsg_to_cv2(data, "rgb8")  
    except CvBridgeError, e:
      print e

    cv_image = numpy.asarray(cv_image)
    #cv2.imshow("Image window", cv_image)
    #cv2.waitKey(1)

    return cv_image
        
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

        # Converting Image, and normalizing
        input_image = convert_image(input_image)
           
        image_rgb = copy.deepcopy(input_image)

        #image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)
        cv2.imshow("cam", image_rgb)
        cv2.waitKey(1)
        
        #image_rgb = cv2.resize(image_rgb, (84, 84), interpolation = cv2.INTER_AREA)
        #image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (84, 84))
        image_rgb = numpy.asarray(image_rgb, dtype=numpy.float32)
        image_rgb /= 127.5
        image_rgb -= 1
        #image_rgb /= 255     
        
        
        #CNN HERE    
        output = model_out.eval(session = sess, feed_dict = {x: [image_rgb], drop_rate: 0.0})
        #output = sess.run([model_out], feed_dict={x: [image_rgb], drop_rate: 1.0})
        print output
        #print output

        x_loc,y_loc, sin_theta, cos_theta = output[0]
        
        label_names = ["x", "y", "sin", "cos"]
        #labels_norm = [2.375, 1.875, 1]
        #labels_norm = [2.4, 1]
        #labels_shift = [2.05, -1.55, 0, 0]
	labels_norm = [2.7435, 1]
	labels_shift = [2.2365, -2.095, 0, 0]
        #min_x = -0.25
        #max_x = 4.5
        #min_y = -3.5
        #max_y = 0.25
        #mean_x = (min_x+max_x)/2
        #mean_y = (min_y+max_y)/2

        x_loc = x_loc * 2.7435 + 2.2365
        y_loc = y_loc * 2.7435 - 2.095
        
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



    x = tf.placeholder(tf.float32, shape=[None, inputHeight, inputWidth, inputChannels])
    y_ = tf.placeholder(tf.float32, shape=[None, nOutput])
    drop_rate = tf.placeholder(tf.float32)
    model_out = model(x, drop_rate)



    # Moving Average Variables
    mean_diffx = 0
    mean_diffy = 0
    mean_total = 0
    index = 0
    counter = 1
    average_image = None
    #model = '/home/borg/amir/yiebo/CNN/sudo/regression/lr0.0001/2/model/iteration1000/model.ckpt'
    #model = '/home/ashantia/University/bach1617/CNN/test/regression/incep/1/model/iteration200/model.ckpt'
    model = '/home/ashantia/2/iteration1000/model.ckpt'
    model = '/home/ashantia/sudo/map/regression/lr0.001/0/model/iteration125/model.ckpt'
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
