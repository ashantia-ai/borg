#!/usr/bin/env python
import cjson
import time
import math

import rospy
from borg_pioneer.srv import *
from nav_msgs.msg import Odometry
import tf

def send_to_memory(dict):
    try:
        resp = service_write(rospy.Time.now(), "odometry", cjson.encode(dict))
    except:
        rospy.logdebug("Memory Unavailable")
        
def odom_report(odom):
    rospy.logdebug("Odometry callback complete")
    
    quat = odom.pose.pose.orientation
    quaternion = [quat.x, quat.y, quat.z, quat.w]
    roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
    
    property_dict = {'x': odom.pose.pose.position.x, 
                     'y': odom.pose.pose.position.y,
                     'angle': math.degrees(yaw),
                     'quaternion': quaternion,
                     'battery_level': 13.0,
                     'time' : time.time()}

    send_to_memory(property_dict)
        
if __name__ == '__main__':
    
    try:
        rospy.init_node('gazebo_controller', anonymous=True)
        
        rospy.loginfo("Waiting for write memory service.")
        service_write = rospy.ServiceProxy('memory', MemorySrv)
        rospy.wait_for_service('memory')
        rospy.loginfo("Write memory found")

        subscriber = rospy.Subscriber("odom", Odometry, odom_report)
        
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
