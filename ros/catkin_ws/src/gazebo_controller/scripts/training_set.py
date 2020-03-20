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
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Point, PoseWithCovarianceStamped, Twist

from os.path import join

#Saving images with gmapping or amcl updated odometry
def gmapping_image(image):
    
    global label
    global data
    global data2
    global hsv_data
    global count 
    
    count += 1
    if count % 100 == 0:
        print count 
    now = rospy.Time(0)
    try:
        transformer.waitForTransform("/map", "/base_link", now, rospy.Duration(0.01))
    except tf.Exception as e:
        print repr(e)
        return
    translation, rotation = transformer.lookupTransform("map", "base_link", now)

    orientation = tf.transformations.euler_from_quaternion(rotation)
    
    new_theta = orientation[2] / (math.pi / 180.0)
    
    odom = {'x': translation[0], 
                 'y': translation[1],
                 'angle': new_theta,
                 'quaternion': rotation,
                 'battery_level': 13.0,
                 'time' : time.time()}
    
    cur_odom = numpy.asarray([odom['x'], odom['y'], math.sin(odom['angle']), math.cos(odom['angle'])])
    try:
        label = numpy.vstack((label, cur_odom))
    except:
        label = cur_odom
    # Converting Image to Grayscale, and normalizing
    image = convert_image(image)
    path = '/home/borg/nav-data/bgrImages/'
    subFolder = 'validation/'
   #$ cv2.imwrite(path + subFolder + str(count) + '.png', image)
   # data2.append(count)
  #  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #  image_gray = cv2.resize(image_gray, shape, interpolation = cv2.INTER_AREA)
  #  image_gray = numpy.asarray(image_gray, dtype=numpy.float32)
  #  image_gray /= 255.
    
    # Converting image to HSV and normalizing
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv = cv2.resize(image_hsv, hsv_shape, interpolation = cv2.INTER_AREA)
    image_hsv = numpy.asarray(image_hsv, dtype=numpy.float32)
    image_hsv[:,:,0] /= 179.
    image_hsv[:,:,1] /= 255.
    image_hsv[:,:,2] /= 255.
    
    try:
    #    data = numpy.vstack((data, numpy.reshape(image_gray, flat_shape)))
        hsv_data = numpy.vstack((hsv_data, numpy.reshape(image_hsv, hsv_flat_shape)))
    except:
    #    data = numpy.reshape(image_gray, flat_shape)
        hsv_data = numpy.reshape(image_hsv, hsv_flat_shape)
def odom_image(odom, image):
    
    # Global variables for saving the gathered grayscale, hsv, and the labeled odometry
    global label
    global data
    global data2
    global hsv_data
    global count 
    
    count += 1
    if count % 100 == 0:
        print count 
    
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
    path = '/home/borg/nav-data/bgrImages/'
    subFolder = 'validation/'
    #cv2.imwrite(path + subFolder + str(count) + '.png', image)
    #data2.append(count)
    #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image_gray = cv2.resize(image_gray, shape, interpolation = cv2.INTER_AREA)
    #image_gray = numpy.asarray(image_gray, dtype=numpy.float32)
    #image_gray /= 255.
    
    # Converting image to HSV and normalizing
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv = cv2.resize(image_hsv, hsv_shape, interpolation = cv2.INTER_AREA)
    image_hsv = numpy.asarray(image_hsv, dtype=numpy.float32)
    image_hsv[:,:,0] /= 179.
    image_hsv[:,:,1] /= 255.
    image_hsv[:,:,2] /= 255.
    
    try:
        #data = numpy.vstack((data, numpy.reshape(image_gray, flat_shape)))
        hsv_data = numpy.vstack((hsv_data, numpy.reshape(image_hsv, hsv_flat_shape)))
    except:
        #data = numpy.reshape(image_gray, flat_shape)
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
    data2 = []
    hsv_data = None
    test_data = None
    label = None
    
    count = 0
    
    hsv_shape = shape = (28, 28) # Destination image shape
    flat_shape = (1, 28 * 28) # Grayscale Flatshape
    hsv_flat_shape = (1, 28 * 28 * 3) # HSV Flat shape
    bridge = CvBridge()
    
    # Path for saving the data
    path = "/home/borg/nav-data/"
    rospy.init_node('validation_set_maker', anonymous=True)
    rospy.loginfo("Node initialized")
    
    # Setting up approximate TimeSynchronizer
    # Warning: You nead ROS indigo message filter package from the core packages
    #odom_sub = message_filters.Subscriber('odom', Odometry)
    '''
    odom_sub = message_filters.Subscriber('amcl_pose', PoseWithCovarianceStamped)
    image_sub = message_filters.Subscriber('/camera/rgb/image_color', Image)
    ts = ApproximateTimeSynchronizer([odom_sub, image_sub], 10, 0.05)
    ts.registerCallback(odom_image)
    '''
    ts = rospy.Subscriber('/camera/rgb/image_color', Image, gmapping_image)
    #ts = rospy.Subscriber('/kinect_color', Image, gmapping_image, queue_size=1, buff_size=1000000)
    transformer = tf.TransformListener()
    

    rospy.loginfo("Callbacks registered")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()
    
    # Saving the data after receiving Keyboard Interrupt
  #  validation_set = (data, label)
    hsv_validation_set = (hsv_data, label)
    
  #  numpy.save(join(path, "real_np_set.npy"), validation_set)
    numpy.save(join(path, "real_hsv_np_set_sunday.npy"), hsv_validation_set)
  #  print len(validation_set)
  #  print len(validation_set[0])
 #   rospy.loginfo("The shapes are %s and %s: ", str(data.shape), str(label.shape))
    rospy.loginfo("The hsv_shapes are %s and %s: ", str(hsv_data.shape), str(label.shape))
    hsv_f = open(join(path, "hsv_temp_set"), 'ab')
    f = open(join(path, "temp_set"), 'ab')
    cPickle.dump(hsv_validation_set, hsv_f)
 #   cPickle.dump(validation_set, f)
    
 #   validation_set2 = (data2, label)
   # validation_set2 = numpy.asarray(validation_set2)
 #   numpy.save(join(path, "real_sift_np_set.npy"), validation_set2)
 #   print len(validation_set2)
 #   print len(validation_set2[0])
  #  hsv_validation_set = (hsv_data, label)
  #  rospy.loginfo("The shapes are %s and %s: ", str(data.shape), str(label.shape))
  #  rospy.loginfo("The hsv_shapes are %s and %s: ", str(hsv_data.shape), str(label.shape))
  #  hsv_f = open(join(path, "hsv_validation_set"), 'ab')
  #  f1 = open(join(path, "sift_temp_set"), 'ab')
  #  cPickle.dump(hsv_validation_set, hsv_f)
 #   cPickle.dump(validation_set2, f1)



