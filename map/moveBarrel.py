#!/usr/bin/env python

import sys
import rospy
import Tkinter as tk
import actionlib
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose

import gazebo_msgs.srv
import gazebo_msgs.msg
import std_srvs.srv

import tf
import math

step = 1

#Modify this to the namespace of gazebo (rosservice list | grep gazebo):
gazebo_ns = "/gazebo"
service_timeout = 10

name_model = "alice"

def move_up(pos):
    pos["y"] += step
    move_to(name_model, pos["x"],pos["y"],0.1, 0,0,0)
    print("moveup")
    
def move_down(pos):
    pos["y"] -= step
    move_to(name_model, pos["x"],pos["y"],0.1, 0,0,0)
    print("movedown")
def move_left(pos):
    pos["x"] -= step
    move_to(name_model, pos["x"],pos["y"],0.1, 0,0,0)
    print("moveleft")
def move_right(pos):
    pos["x"] +=step
    move_to(name_model, pos["x"],pos["y"],0.1, 0,0,0)
    print("moveright")

def move_to(model, x, y, z, roll, pitch, yaw):
    """
    Moves a model to the specified location and orientation.
    @param  model   The name of the model.
    """
    service = '%s/set_model_state' % gazebo_ns
    rospy.wait_for_service(service, timeout = service_timeout)
    set_model_state = rospy.ServiceProxy(service, gazebo_msgs.srv.SetModelState)
    msg = gazebo_msgs.msg.ModelState()
    msg.model_name = model
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z

    quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    msg.pose.orientation.x = quat[0]
    msg.pose.orientation.y = quat[1]
    msg.pose.orientation.z = quat[2]
    msg.pose.orientation.w = quat[3]
    msg.reference_frame = "world"

    set_model_state(msg)

def get_position(name):
    """
    #param      The name of the model.
    @return     The position as (x, y, yaw).
    """
    service = '%s/get_model_state' % gazebo_ns
    rospy.wait_for_service(service, timeout = service_timeout)
    get_model_state = rospy.ServiceProxy(service, gazebo_msgs.srv.GetModelState)
    response = get_model_state(model_name = name)
    pos = {}
    #x and y:
    pos["x"] = response.pose.position.x
    pos["y"] = response.pose.position.y
    #yaw:
    quaternion = (
            response.pose.orientation.x,
            response.pose.orientation.y,
            response.pose.orientation.z,
            response.pose.orientation.w)
    pos["yaw"] = tf.transformations.euler_from_quaternion(quaternion)[2]
    return pos



def onKeyPress(event):
    text.insert('end', 'You pressed %s\n' % (event.char, ))
    pos = get_position(name_model)
    c = event.char
    if(c == 'a'):
        move_up(pos)
    if(c == 'd'):
        move_down(pos)
    if(c == 's'):
        move_left(pos)
    if(c == 'w'):
        move_right(pos)
   



def moveBarrel(root):
    # Creates the SimpleActionClient, passing the type of the action
    # (FibonacciAction) to the constructor.
#name,x,y,angle
#kitchen,-1.903,4.042,90.0
#dining_table,1.248,0.682,0.0
#work_space,2.743,-2.201,-45.0
#living_room,3.212,-7.414,-90

    root.mainloop()

    

if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
        root = tk.Tk()
        root.geometry('300x200')
        text = tk.Text(root, background='black', foreground='white', font=('Comic Sans MS', 12))
        text.pack()
        root.bind('<KeyPress>', onKeyPress)

        rospy.init_node('moveBarrel_py')
        moveBarrel(root)
    except rospy.ROSInterruptException:
        print "program interrupted before completion"

