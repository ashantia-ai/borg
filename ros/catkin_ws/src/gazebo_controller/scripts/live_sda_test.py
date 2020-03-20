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
        
def live_test(orig_image):
    #print label.shape
    orig_image = convert_image(orig_image)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(orig_image, (28, 28), interpolation = cv2.INTER_AREA)
    image = numpy.asarray(image, dtype = numpy.float32)
    image = numpy.reshape(image, (1, 28 * 28))
    image /= 255
    
    normal_img = (image - numpy.mean(image)) / numpy.std(image)
    
    res = get_f(image)
    
    normal_res = (res - numpy.mean(res)) / numpy.std(res)
    
    res = numpy.reshape(res, (28,28))
    res2 = cv2.resize(res, (320, 240), interpolation = cv2.INTER_LINEAR)
    res2 *= 255
    res2 = numpy.asarray(res2, dtype = numpy.uint8)
    
    #res2 = cv2.equalizeHist(res2)
    #print numpy.max(res2)
    #print numpy.min(res2)
    
    diff = normal_img - normal_res
    #print "normal Img mean: ", numpy.mean(normal_img), "\t std: ", numpy.var(normal_img)
    #print "res Img mean: ", numpy.mean(normal_res), "\t std: ", numpy.var(normal_res)
    
    translate = diff
    #translate = (translate - numpy.mean(translate)) / numpy.std(translate)
    #print "diff Img mean: ", numpy.mean(translate), "\t std: ", numpy.var(translate)
    #translate = (translate * 127 * numpy.std(translate) + 127)
    #translate = numpy.asarray(translate, dtype = numpy.uint8)
    #print "translate Img mean: ", numpy.mean(translate), "\t std: ", numpy.var(translate)
    translate = numpy.reshape(translate, (28,28))
    translate = cv2.resize(translate, (320, 240), interpolation = cv2.INTER_LINEAR)
    
    sift = cv2.SIFT()
    kp = sift.detect(orig_image)
    print "angle ", kp[0].angle
    print "size ",kp[0].size
    print "point ",kp[0].pt
    print "pctave",kp[0].octave
    print "response ", kp[0].response
    cv2.imshow("Original Image", orig_image)
    cv2.imshow("Reconstructed Image", res2)
    cv2.imshow("Difference Image, 28x28", translate)
    #cv2.imshow("Reconstructed Image, 28x28", res)
    cv2.waitKey(1)

def live_test_hsv(orig_image):
    #print label.shape
    orig_image = convert_image(orig_image)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV)
    
    image = cv2.resize(orig_image, (28, 28), interpolation = cv2.INTER_AREA)
    image = numpy.asarray(image, dtype = numpy.float32)
    image[:,:,0] /= 179
    image[:,:,1] /= 255
    image[:,:,2] /= 255

    image = numpy.reshape(image, (1, 28 * 28 * 3))
        
    normal_img = (image - numpy.mean(image)) / numpy.std(image)
    
    #print image.size
    res = get_f(image)
    
    normal_res = (res - numpy.mean(res)) / numpy.std(res)
    
    res = numpy.reshape(res, (28,28, 3))
    res2 = cv2.resize(res, (320, 240), interpolation = cv2.INTER_LINEAR)
    #res2 *= 255
    #res2 = numpy.asarray(res2, dtype = numpy.uint8)
    
    #res2 = cv2.equalizeHist(res2)
    #print numpy.max(res2)
    #print numpy.min(res2)
    
    diff = normal_img - normal_res
    #print "normal Img mean: ", numpy.mean(normal_img), "\t std: ", numpy.var(normal_img)
    #print "res Img mean: ", numpy.mean(normal_res), "\t std: ", numpy.var(normal_res)
    
    #translate = diff
    #translate = (translate - numpy.mean(translate)) / numpy.std(translate)
    #print "diff Img mean: ", numpy.mean(translate), "\t std: ", numpy.var(translate)
    #translate = (translate * 127 * numpy.std(translate) + 127)
    #translate = numpy.asarray(translate, dtype = numpy.uint8)
    #print "translate Img mean: ", numpy.mean(translate), "\t std: ", numpy.var(translate)
    #translate = numpy.reshape(translate, (28,28))
    #translate = cv2.resize(translate, (320, 240), interpolation = cv2.INTER_LINEAR)
    
    sift = cv2.SIFT()
    kp = sift.detect(orig_image)
    
    orig_rgb = cv2.cvtColor(orig_image, cv2.COLOR_HSV2BGR)
    res2[:,:,0] *= 179
    res2[:,:,1] *= 255
    res2[:,:,2] *= 255
    res2 = numpy.asarray(res2, dtype = numpy.uint8)
    
    res2_rgb = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
    cv2.imshow("Original Image", orig_rgb)
    cv2.imshow("Reconstructed Image", res2_rgb)
    #cv2.imshow("Difference Image, 28x28", translate)
    #cv2.imshow("Reconstructed Image, 28x28", res)
    cv2.waitKey(1)

    
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
    model = '/media/data/nav-data/trainedModels/model2014-07-05_19:20:53'
    model = '/media/data/nav-data/june-2014w/compressed/model2014-07-15_09:52:43'
    model = '/home/sim-borg/NN/pics3/model/model__hsv_2_750_2015-04-24_14:25:35'
    x = T.fmatrix('x')
    f = open(model, 'rb')
    model = cPickle.load(f)
    out = model.get_reconstructed_input(x)
    #out = model.encode(x)
    get_f = theano.function([x], out)
    
    shape = (28, 28, 3)
    flat_shape = (1, 28 * 28 * 3)
    bridge = CvBridge()
    
    path = "/media/data/nav-data/"
    rospy.init_node('live_sda_test', anonymous=True)
    rospy.loginfo("Node initialized")
    
    rospy.Subscriber("/camera/rgb/image_color", Image, live_test_hsv)
    rospy.loginfo("Callbacks registered")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()



