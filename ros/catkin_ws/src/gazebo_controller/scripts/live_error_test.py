#!/usr/bin/env python
import cPickle, sys
import math, numpy, time, cv2
import sd_autoencoder

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

from pyfann import libfann

import message_filters
from message_filters import ApproximateTimeSynchronizer

from linear_regression import NNRegression, LinearRegression

def rescale(scale):    
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

def live_test(odom, orig_image):
    global mean_diffx
    global mean_diffy
    global mean_total
    global index
    #print label.shape
    orig_image = convert_image(orig_image)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(orig_image, (28, 28), interpolation = cv2.INTER_AREA)
    image = numpy.asarray(image, dtype = numpy.float32)
    image = numpy.reshape(image, (1, 28 * 28))
    image /= 255
    
    normal_img = (image - numpy.mean(image)) / numpy.std(image)

 #   res = get_f(image)
    output = get_f(image)  # encode 
    #get_nn = output
    output = ann.run(output[0]) # output of network
    normOutput = output
    print 'Norm out:', normOutput[0], ', y: ', normOutput[1], ', sin: ', normOutput[2], ', cos: ', normOutput[3]
    output = rescale(output)
    
    
    quat = odom.pose.pose.orientation
    quaternion = [quat.x, quat.y, quat.z, quat.w]
    roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
    
    odom = {'x': odom.pose.pose.position.x, 
                     'y': odom.pose.pose.position.y,
                     'angle': yaw,
                     'quaternion': quaternion,
                     'battery_level': 13.0,
                     'time' : time.time()}
    
    cur_odom = numpy.asarray([odom['x'], odom['y'], math.sin(odom['angle']), math.cos(odom['angle'])])
   
    print 'Network: x: ', output[0], ', y: ', output[1], ', sin: ', output[2], ', cos: ', output[3]
    print 'Odom:  : ', cur_odom
    
    xDiff = abs(odom['x'] - output[0])
    yDiff = abs(odom['y'] - output[1])
    sDiff = abs(math.sin(odom['angle']) - output[2])
    cDiff = abs(math.cos(odom['angle']) - output[3])
    print 'Diff x: ', xDiff, ' y: ', yDiff, ', sin: ', sDiff, ', cos: ', cDiff
    
    
    mean_diffx = (xDiff + index * mean_diffx) / (index + 1)
    mean_diffy = (yDiff + index * mean_diffy) / (index + 1)
    mean_total = (math.sqrt(yDiff**2 + xDiff **2) + index * mean_total) / (index + 1)
    index += 1
    
    print "Moving Average xDiff: ", mean_diffx, " yDiff: ", mean_diffy, " total: ", mean_total
    print ' '
    
  #  model.x = image;
    
#    print model.top_linear_layer.y_pred
#     normal_res = (res - numpy.mean(res)) / numpy.std(res)
#     
#     res = numpy.reshape(res, (28,28))
#     res2 = cv2.resize(res, (320, 240), interpolation = cv2.INTER_LINEAR)
#     res2 *= 255
#     res2 = numpy.asarray(res2, dtype = numpy.uint8)
#         
#     diff = normal_img - normal_res
#     translate = diff
#     
#     
#     translate = numpy.reshape(translate, (28,28))
#     translate = cv2.resize(translate, (320, 240), interpolation = cv2.INTER_LINEAR)
#     
#     cv2.imshow("Original Image", orig_image)
#     cv2.imshow("Reconstructed Image", res2)
#     cv2.imshow("Difference Image, 28x28", translate)
#     cv2.imshow("Reconstructed Image, 28x28", res)
#     cv2.waitKey(1)


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
 #   fine_tune = '/home/rik/nav-data/NNtest'
 #   f = open(fine_tune, 'rb')
 #   f_model = cPickle.load(f)
 #   f.close()
    # Moving Average Variables
    mean_diffx = 0
    mean_diffy = 0
    mean_total = 0
    index = 0
    
    model = '/home/rik/nav-data/june-2014w/compressed/model2014-07-18_20:13:41'
  #  networkPath = '/home/rik/nav-data/networks/network.net2014-07-20_03:18:44'
    networkPath = '/home/rik/sudo/brain/src/vision/autoencoder/network.net2014-07-21_17:50:23'
   # model = '/home/rik/nav-data/june-2014w/compressed/model2014-07-15_09:52:43'
    x = T.fmatrix('x')
    f = open(model, 'rb')
    model = cPickle.load(f)
  #  out = model.get_reconstructed_input(x)
    out = model.encode(x)
    #out = model.encode(x)
 #   calc = f_model.calc()
    get_f = theano.function([x], out)   
    
  #  get_nn = theano.function([x], calc)
    
    ann = libfann.neural_net()
    ann.create_from_file(networkPath)
    shape = (28, 28, 1)
    flat_shape = (1, 28 * 28)
    bridge = CvBridge()
    
    path = "/home/rik/nav-data/"
    rospy.init_node('live_error_test', anonymous=True)
    rospy.loginfo("Node initialized")
   
    odom_sub = message_filters.Subscriber('odom', Odometry)
    image_sub = message_filters.Subscriber('/sudo/bottom_webcam/image_raw', Image)
    
    ts = ApproximateTimeSynchronizer([odom_sub, image_sub], 100, 0.2)   
    
    ts.registerCallback(live_test)
    
    rospy.loginfo("Callbacks registered")   
    
   
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()