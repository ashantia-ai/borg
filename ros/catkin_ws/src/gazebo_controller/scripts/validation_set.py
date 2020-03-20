#!/usr/bin/env python
import cPickle, sys
import math, numpy, time, cv2

import rospy, roslib
from nav_msgs.msg import Odometry
import tf

import message_filters
from message_filters import ApproximateTimeSynchronizer

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

from os.path import join
        
def odom_image(odom, image):
    
    # Global variables for saving the gathered grayscale, hsv, and the labeled odometry
    global label
    global data
    global hsv_data
    
    quat = odom.pose.pose.orientation
    quaternion = [quat.x, quat.y, quat.z, quat.w]
    roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
    # Warning, The angle should be in radian. Yaw is by default Radian
    odom = {'x': odom.pose.pose.position.x, 
                     'y': odom.pose.pose.position.y,
                     'angle': yaw,
                     'quaternion': quaternion,
                     'battery_level': 13.0,
                     'time' : time.time()}
    
    cur_odom = numpy.asarray([odom['x'], odom['y'], math.sin(odom['angle']), math.cos(odom['angle'])])
    try:
        label = numpy.vstack((label, cur_odom))
    except:
        label = cur_odom
    # Converting Image to Grayscale, and normalizing
    image = convert_image(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.resize(image_gray, shape, interpolation = cv2.INTER_AREA)
    image_gray = numpy.asarray(image_gray, dtype=numpy.float32)
    image_gray /= 255
    
    # Converting image to HSV and normalizing
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv = cv2.resize(image_hsv, hsv_shape, interpolation = cv2.INTER_AREA)
    image_hsv = numpy.asarray(image_hsv, dtype=numpy.float32)
    image_hsv[:,:,0] /= 179
    image_hsv[:,:,1] /= 255
    image_hsv[:,:,2] /= 255
    
    try:
        data = numpy.vstack((data, numpy.reshape(image_gray, flat_shape)))
        hsv_data = numpy.vstack((hsv_data, numpy.reshape(image_hsv, hsv_flat_shape)))
    except:
        data = numpy.reshape(image_gray, flat_shape)
        hsv_data = numpy.reshape(image_hsv, hsv_flat_shape)
# Convert image from ROS format to OpenCV
def convert_image(data):
    try:
      cv_image = bridge.imgmsg_to_cv2(data, "bgr8")  
    except CvBridgeError, e:
      print e

    cv_image = numpy.asarray(cv_image)
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(1)

    return cv_image


if __name__ == '__main__':
    validation_set = None
    data = None
    hsv_data = None
    test_data = None
    label = None
    
    hsv_shape = shape = (28, 28) # Destination image shape
    flat_shape = (1, 28 * 28) # Grayscale Flatshape
    hsv_flat_shape = (1, 28 * 28 * 3) # HSV Flat shape
    bridge = CvBridge()
    
    # Path for saving the data
    path = "/media/data/nav-data/"
    rospy.init_node('validation_set_maker', anonymous=True)
    rospy.loginfo("Node initialized")
    
    # Setting up approximate TimeSynchronizer
    # Warning: You nead ROS indigo message filter package from the core packages
    odom_sub = message_filters.Subscriber('odom', Odometry)
    image_sub = message_filters.Subscriber('/sudo/bottom_webcam/image_raw', Image)
    ts = ApproximateTimeSynchronizer([odom_sub, image_sub], 100, 0.2)
    
    
    ts.registerCallback(odom_image)
    rospy.loginfo("Callbacks registered")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()
    
    # Saving the data after receiving Keyboard Interrupt
    validation_set = (data, label)
    hsv_validation_set = (hsv_data, label)
    rospy.loginfo("The shapes are %s and %s: ", str(data.shape), str(label.shape))
    rospy.loginfo("The hsv_shapes are %s and %s: ", str(hsv_data.shape), str(label.shape))
    hsv_f = open(join(path, "hsv_validation_set"), 'ab')
    f = open(join(path, "validation_set"), 'ab')
    cPickle.dump(hsv_validation_set, hsv_f)
    cPickle.dump(validation_set, f)



