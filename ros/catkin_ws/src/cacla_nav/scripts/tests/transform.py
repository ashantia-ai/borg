'''
Gets transform from Xtion to Baselink from Alice and print it
'''

import rospy
import math
import tf
import geometry_msgs.msg

def get_radian(data): 
        
    degrees = tf.transformations.euler_from_quaternion(data)
    return degrees

if __name__ == "__main__":
    rospy.init_node('Transform_node')
    
    listener = tf.TransformListener()
    
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform("front_rgb_optical_link", "base_footprint", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        
        print "Translation between frames: " , trans
        print "Rotation between frames:", get_radian(rot)
        angular = 4 * math.atan2(trans[1], trans[0])
        linear = 0.5 * math.sqrt(trans[0] ** 2 + trans[1] ** 2)
        
        rate.sleep()