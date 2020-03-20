#!/usr/bin/env python
import cPickle, sys
import math, numpy, time, cv2
import sd_autoencoder

from operator import itemgetter, attrgetter

import rospy, roslib
from nav_msgs.msg import Odometry
import tf

import theano
from theano import tensor as T, config, shared
from theano.tensor.shared_randomstreams import RandomStreams

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

from os.path import join

#from pyfann import libfann

import message_filters
#from message_filters import ApproximateTimeSynchronizer

from linear_regression import NNRegression, LinearRegression

from geometry_msgs.msg import PoseArray, Pose

from multiprocessing import Lock

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
    maxX = 7.3
    minY = -1.9
    maxY = 2.57 + -minY
    
    scale[0] *= maxX
    scale[1] *= maxY
    scale[1] -= -minY
    
    scale[2] *= 2
    scale[2] -= 1
    scale[3] *= 2
    scale[3] -= 1
        
    return scale
def rescale(scale):    
    minX = -2.8
    maxX = 10.90 + -minX
    
    minY = -5.8
    maxY = 9.90 + -minY
    
    # normalize the x 
    scale[0] *= maxX
    scale[0] -= -minX
    
    
    #normalize the y
    scale[1] *= maxY
    scale[1] -= -minY
    
    
    # normalize the sin and cos by adding 1 and divide by 2
    scale[2] += 1
    scale[2] /= 2
    scale[3] += 1
    scale[3] /= 2
    
    scale[2] *= 2
    scale[2] -= 1
    scale[3] *= 2
    scale[3] -= 1
    
    return scale


class Tester(object):
    def __init__(self):
        self.transformer = tf.TransformListener()
        self.publisher = rospy.Publisher('estimated_pose', PoseArray)
        
        self.location_array = PoseArray()
        self.location_array.header.frame_id="odom"
        self.lock = Lock()
        
        self.counter = 0
        
    def gmapping_image(self, image):
        global mean_diffx
        global mean_diffy
        global mean_total
        global index
        global average_image
        global counter
        
        
        # Converting Image to Grayscale, and normalizing
        image = convert_image(image)
           
        # Converting image to HSV and normalizing
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_hsv_cp = image_hsv.copy()
        image_hsv = cv2.resize(image_hsv, (28, 28), interpolation = cv2.INTER_AREA)
        image_hsv = numpy.asarray(image_hsv, dtype=numpy.float32)
        
        image_hsv[:,:,0] /= 179.
        image_hsv[:,:,1] /= 255.
        image_hsv[:,:,2] /= 255.
        image = numpy.asarray(image_hsv, dtype = numpy.float32)
        image = numpy.reshape(image_hsv, (1, 28 * 28 * 3))
        
            
        output = get_f(image[:,:])
        
        x,y, sin_theta, cos_theta = scale(output[0])
        theta = math.atan2(cos_theta,sin_theta)
        theta = math.atan2(sin_theta, cos_theta)
        
        somePose = Pose()
        
        somePose.position.x = x
        somePose.position.y = y
        somePose.position.z = 1.0

        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta, 'sxyz')
        
        somePose.orientation.x = quaternion[0]
        somePose.orientation.y = quaternion[1]
        somePose.orientation.z = quaternion[2]
        somePose.orientation.w = quaternion[3]
        #self.lock.acquire()
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
    

if __name__ == '__main__':
   # fine_tune = '/media/data/nav-data/best_model'
   # f = open(fine_tune, 'rb')
   # f_model = cPickle.load(f)
   # f.close()
    # Moving Average Variables
    mean_diffx = 0
    mean_diffy = 0
    mean_total = 0
    index = 0
    counter = 1
    average_image = None
  #  model = '/media/data/nav-data/june-2014w/compressed/model2014-07-15_09:52:43'
    
    model = '/home/amir/sudo/ros/catkin_ws/src/gazebo_controller/scripts/model__hsv__best_phase2_full_[750]'
    
    x = T.fmatrix('x')
    y = T.fmatrix('y')
    f = open(model, 'rb')
    model = cPickle.load(f)
    nn_output = model.get_output(x)
    reconstruct = model.get_reconstructed_input(x)

 
    get_f = theano.function([x], outputs = nn_output)
    get_r = theano.function([x], outputs = reconstruct)
    #theano.function(inputs, outputs, mode, updates, givens, no_default_updates, accept_inplace, name, rebuild_strict, allow_input_downcast, profile, on_unused_input)   
    shape = (28, 28)
    flat_shape = (1, 28 * 28 * 3)
    bridge = CvBridge()
    
    
    rospy.init_node('live_error_test', anonymous=True)
    rospy.loginfo("Node initialized")
    
    #Initializing the testing class
    test_hsv = Tester()
    
    ts = rospy.Subscriber('/sudo/bottom_webcam/image_raw', Image, test_hsv.gmapping_image)
    rospy.loginfo("Callbacks registered")   
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()
