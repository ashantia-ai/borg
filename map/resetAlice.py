#!/usr/bin/env python

import sys
import rospy
import Tkinter as tk
import actionlib
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import SpawnModel
import geometry_msgs
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseWithCovariance, Pose, Point, Quaternion

import gazebo_msgs.srv
import gazebo_msgs.msg
import std_srvs.srv

import tf
import math

#Modify this to the namespace of gazebo (rosservice list | grep gazebo):
gazebo_ns = "/gazebo"
service_timeout = 10

name_model = "alice"


def reset():
    """
    Moves a model to the specified location and orientation.
    @param  model   The name of the model.
    """
    service = '%s/set_model_state' % gazebo_ns
    rospy.wait_for_service(service, timeout = service_timeout)
    set_model_state = rospy.ServiceProxy(service, gazebo_msgs.srv.SetModelState)
    msg = gazebo_msgs.msg.ModelState()
    msg.model_name = name_model

    #quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    msg.pose =   Pose(Point(0,0, 0.190), Quaternion(0.000, 0.000, 0.223, 0.975))
    msg.reference_frame = "world"
   
    set_model_state(msg)
    setestimate()

def setestimate():
    pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size = 1)
    p = PoseWithCovarianceStamped()
    msg = PoseWithCovariance()
    msg.pose =  Pose(Point(0,0, 0.190), Quaternion(0.000, 0.000, 0.223, 0.975))

    msg.covariance = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0];
    
    p.pose = msg
    rate = rospy.Rate(10)   

    index = 0
    while index < 10: # This is hacky hacky Rik 24-11-2015
        index += 1
        pub.publish(p)
        rate.sleep()

    

if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
    
        rospy.init_node('resetAlice_py')
        reset()
    except rospy.ROSInterruptException:
        print "program interrupted before completion"

