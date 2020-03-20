#!/usr/bin/env python
import argparse
import cPickle, sys

import math, numpy, time, cv2

from operator import itemgetter, attrgetter

import rospy, roslib
from nav_msgs.msg import Odometry
import tf

import tensorflow as tfg
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

from os.path import join

#from pyfann import libfann

import message_filters
#from message_filters import ApproximateTimeSynchronizer

from geometry_msgs.msg import PoseArray, Pose

from multiprocessing import Lock

import keras
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from lenet import LeNet

from math import fabs
def convert_image(data):
    try:
      cv_image = bridge.imgmsg_to_cv2(data, "bgr8")  
    except CvBridgeError, e:
      print e

    cv_image = numpy.asarray(cv_image)
    #cv2.imshow("Image window", cv_image)
    #cv2.waitKey(1)

    return cv_image
        
def scale(labels):
    '''
    minX = -3.5    
    maxX = 4.75

    minY = -10.25
    maxY = 10.25
    
    scale[0] = (scale[0]) * maxX
    scale[1] = (scale[1]) * maxY
        
    '''

    
    minX = -0.25
    maxX = 4.5 + fabs(minX)
    
    labels[0] *= maxX
    labels[0] -= fabs(minX)
    
    
    minY = -3.5
    maxY = 0.25 + fabs(minY)
    
    labels[1] *= maxY
    labels[1] -= fabs(minY)
    
    labels[2] *= 2
    labels[2] -= 1
    
    labels[3] *= 2
    labels[3] -= 1
    
    return labels
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
        self.transformer = tf.TransformListener()
        self.publisher = rospy.Publisher('estimated_pose_list', PoseArray, queue_size = 1)
        self.pose_publisher = rospy.Publisher('estimate_pose', Pose, queue_size = 1)
        
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
        #image = cv2.imread('/home/amir/sudo/map/sudo_data_set/63/63_-500.jpg')  
        # Converting image to HSV and normalizing
        #image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = image
        image_hsv = cv2.resize(image_hsv, (84, 84))
        image_hsv = numpy.asarray(image_hsv, dtype=numpy.float32)
        
        
        image_hsv /= 255.
        
        image_hsv = numpy.reshape(image_hsv, (1,84,84,3))
        #image_hsv = numpy.asarray(image_hsv, dtype = numpy.float32)
        
        global graph
        output = None
        with graph.as_default():
            #print image_hsv.min()
            #print image_hsv.max()
            #print image_hsv.mean()
            
            output = model_MatFra.predict(image_hsv)
            #print output
        #output = get_f(image[:,:])
        
        x,y, sin_theta, cos_theta = scale(output[0])
        
        #theta = math.atan2(cos_theta,sin_theta)
        theta = math.atan2(sin_theta, cos_theta)
        #print theta
        
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
    


if __name__ == '__main__':
    
    config = tfg.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tfg.Session(config=config)

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save-model", type=int, default=-1, help="(optional) whether or not model should be saved to disk")
    ap.add_argument("-l", "--load-model", type=int, default=-1, help="(optional) whether or not pre-trained model should be loaded")
    ap.add_argument("-w", "--weights", type=str, help="(optional) path to weights file")
    args = vars(ap.parse_args())

    #Param Configuration!    

    path = '/home/amir/CNN/'
    log_path = '/home/amir/logs/'
    pic_shape = (84, 84, 3)
    opt = SGD(lr=0.0001)
    n_outputs = 4
    train = True  # If false, load parameters and run validation!

    precise_evaluation = True
    
    print "Running Experiment: "


    tbCallBack = keras.callbacks.TensorBoard(log_dir=log_path, 
                            histogram_freq=0, write_graph=True, write_images=False)

    print("[INFO] compiling model...")
    print 
    model_MatFra = LeNet.build(pic_shape[0],pic_shape[1],pic_shape[2], outputs=n_outputs,
             mode=1, weightsPath=args["weights"] if args["load_model"] > 0 else None)
    model_MatFra.compile(loss="mean_squared_error", optimizer=opt , metrics=["accuracy"])

    print model_MatFra.summary()
    
    model_MatFra.load_weights("/home/borg/SabBido/NN_param/checkpoint")
    graph = tfg.get_default_graph()
               
    bridge = CvBridge()
    
    
    rospy.init_node('live_error_test', anonymous=True)
    rospy.loginfo("Node initialized")
    
    #Initializing the testing class
    test_hsv = Tester()
    
    ts = rospy.Subscriber("/sudo/bottom_webcam/image_raw", Image, test_hsv.gmapping_image)
    rospy.loginfo("Callbacks registered")   
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()
