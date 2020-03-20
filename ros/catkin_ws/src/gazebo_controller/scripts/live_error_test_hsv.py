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

from pyfann import libfann

import message_filters
from message_filters import ApproximateTimeSynchronizer

from linear_regression import NNRegression, LinearRegression

def extract_features(image):
   
    # init sift and compute keypoints
    sift = cv2.SIFT()
    kp = sift.detect(image, None) 
    kp,des = sift.compute(image,kp)
    
    nFeat = []
    for idx in range(0, len(kp)):
        nFeat.append([kp[idx].response, kp[idx].pt, des[idx]]) 
    
    nFeat.sort(key=itemgetter(0))
    nFeat.reverse()
    
    if (len(kp) >= 4): # check if at least 4 keypoints where found 
        sizeKp = 4 # search for the 4 keypoints
    else: 
        sizeKp = len(kp)  # else use the keypoints (<4) that are available
    
    positions = [] # get the position (in the color image)            
         
    ordered = False
    while (not ordered):
        check = True # check if array is ordered 
        
        for idx in range(0, sizeKp - 1):
            y, x = nFeat[idx][1]
            y2, x2 = nFeat[idx + 1][1]
            if (y == y2 and x > x2): # same row, but x[i] > x[i+1]
                # switch positions
                nFeat[idx], nFeat[idx + 1] = nFeat[idx + 1], nFeat[idx]
                check = False
            
            elif (y > y2): # idx is lower then idx + 1               
                # switch positions
                nFeat[idx], nFeat[idx + 1] = nFeat[idx + 1], nFeat[idx]
                check = False
                
        if (check == True): # array is sorted correctly
            ordered = True       
    # end while        
    feature = []  
    featureTemp = []       
    
    for idx in range(0, sizeKp): # Take the normalized histograms as features
        nFeat[idx][2] /= 255.
        featureTemp.append(nFeat[idx][2])
        #feature = np.hstack(nFeat[idx][2])
        
           
    if (sizeKp != 4): # if not enough keypoints found, add zero points 
        for idx in range(sizeKp, 4):
            zeroFeature = np.zeros(128)
            featureTemp.append(zeroFeature)
         #   feature = np.hstack(zeroFeature)
      
    feature = numpy.hstack(featureTemp)   
    return feature

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

def gmapping_image(image):
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
    
    if counter % 4 != 0:
        try:
            average_image = numpy.add(average_image, image) / 2
        except:
            average_image = image
        counter += 1
        return
    else:
        counter = 1
        print average_image.shape
        image = average_image
    
    transformer.waitForTransform("/map", "/base_link", rospy.Time(0), rospy.Duration(50))
    translation, rotation = transformer.lookupTransform("map", "base_link", rospy.Time(0))

    orientation = tf.transformations.euler_from_quaternion(rotation)
    
    new_theta = orientation[2] / (math.pi / 180.0)
    new_odo = {"x": translation[0] * 1000.0,
               "y": translation[1] * 1000.0, 
               "angle": new_theta}
    
    odom = {'x': translation[0], 
                 'y': translation[1],
                 'angle': new_theta,
                 'quaternion': rotation,
                 'battery_level': 13.0,
                 'time' : time.time()}
    
    cur_odom = numpy.asarray([odom['x'], odom['y'], math.sin(odom['angle']), math.cos(odom['angle'])])

    
    
   # feature = []
   # feature1 = []
   # feature1 = extract_features(image_hsv_cp)
   # feature1 = numpy.reshape(feature1, (1, 128*4))
   # print feature1.shape
   # feature.append(feature1)
   # print image.shape
   # feature.append(image)
   # feature = numpy.hstack(feature)
   #output = get_f(feature[:,:])  # encode 
    output = get_f(image[:,:])
 #   output = get_nn(output)[0]
    output = ann.run(output[0]) # output of network
    normOutput = output
    
    print 'Norm out:', normOutput[0], ', y: ', normOutput[1], ', sin: ', normOutput[2], ', cos: ', normOutput[3]
    output = rescale(output)
    
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
    

def live_test(odom, orig_image):
    global mean_diffx
    global mean_diffy
    global mean_total
    global index
    #print label.shape
    orig_image = convert_image(orig_image)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV)
    image = cv2.resize(orig_image, (28, 28), interpolation = cv2.INTER_AREA)
    image = numpy.asarray(image, dtype = numpy.float32)
    image[:,:,0] /= 179
    image[:,:,1] /= 255
    image[:,:,2] /= 255
    image = numpy.asarray(image, dtype = numpy.float32)
    image = numpy.reshape(image, (1, 28 * 28 * 3))
    
    
    normal_img = (image - numpy.mean(image)) / numpy.std(image)

 #   res = get_f(image)
    output = get_f(image[:,:])  # encode 
   # output = get_nn(output)[0]
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
    
    model = '/home/borg/nav-data/realModels/model__hsv_40_[750]_2014-09-09_180421'
    
  #  networkPath = '/home/rik/nav-data/networks/network.net2014-07-20_03:18:44'
    networkPath = '/home/borg/nav-data/networks/realmodel/new/model4/network_7200_2014-09-13_23:34:01.net'
    
    
    # x, y mean of 0.80cm 
  #  networkPath = '/home/borg/nav-data/networks/realmodel/new/model4/network_7200_2014-09-13_23:34:01.net'
   # model = '/home/rik/nav-data/june-2014w/compressed/model2014-07-15_09:52:43'
    x = T.fmatrix('x')
    y = T.fmatrix('y')
    f = open(model, 'rb')
    model = cPickle.load(f)
  #  out = model.get_reconstructed_input(x)
    out = model.encode(x)
  #  calc = f_model.calc(x)
    
    get_f = theano.function([x], out)   
 #   get_nn = theano.function([x], calc, allow_input_downcast=True)
    
    ann = libfann.neural_net()
    ann.create_from_file(networkPath)
    shape = (28, 28)
    flat_shape = (1, 28 * 28 * 3)
    bridge = CvBridge()
    
    
    rospy.init_node('live_error_test', anonymous=True)
    rospy.loginfo("Node initialized")
    '''
    odom_sub = message_filters.Subscriber('odom', Odometry)
    image_sub = message_filters.Subscriber('/sudo/bottom_webcam/image_raw', Image)
    
    ts = ApproximateTimeSynchronizer([odom_sub, image_sub], 100, 0.2)   
    
    ts.registerCallback(live_test)
    '''
    ts = rospy.Subscriber('/kinect_color', Image, gmapping_image)
    transformer = tf.TransformListener()
    
    rospy.loginfo("Callbacks registered")   
    
   
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()