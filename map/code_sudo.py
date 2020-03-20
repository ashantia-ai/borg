#!/usr/bin/python
import sys
import os
import errno
import threading
import signal  ##for handling OS signals (e.g. ctrl+c)
import math
import numpy
import cv2
import yaml
import rospy
import time
import tf
from tqdm import tqdm

from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_msgs.msg import Header, Float64
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Twist, Vector3
from control_msgs.msg import JointControllerState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


import cPickle

'''
This program creates a data set from map file. The Move_Base should be running for this program to work.
It will create a yaml file, and a planned file.
'''
if __name__ ==  '__main__' :
    rospy.init_node("planner")


    availableLoc = []
    mapVar = yaml.load(open("kitchen.yaml", 'r'))
    map = cv2.imread(mapVar['image'],0)

    height = map.shape[0]
    width = map.shape[1]
    resolution = mapVar['resolution']
    origin = mapVar['origin']
    print "map: sim_map, height {}, width {}".format(height, width)

# # #-------------------------------------------------------------------------

    rospy.wait_for_service('move_base/make_plan')
    try:
        request = rospy.ServiceProxy('move_base/make_plan', GetPlan)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


    print "got service 'move_base/make_plan'"
    startPos = Point(0, 0, 0.000)
    startOrientation = Quaternion(0,0, 0.00, 1)
    # startPose = Pose()
    startPose = Pose(startPos, startOrientation)
    startHead = Header(0, rospy.Time.now(), "map")
    start = PoseStamped(header = startHead, pose = startPose)

    #TODO:change orientaion shit load of time. it doesn't have it
    availableLoc = []
    try:
        f = open('map_positions', 'rb')
        availableLoc = cPickle.load(f)
        print "loaded map positions"
    except:
        print "No file to load, calculating map poses"

        w = 5
        while w < width :
            h = 5
            while h < height :
                angle = int(0)
                available = True
                loc_angle = []
                while angle < 360:
                    quat=tf.transformations.quaternion_from_euler(0, 0, math.radians(angle))
                    goalPos = Point(w*resolution+origin[0], h*resolution+origin[1], 0.01)
                    # print "checking {}".format(goalPos)
                    goalOrientation = Quaternion(quat[0], quat[1], quat[2], quat[3])
                    pose = Pose(position = goalPos, orientation = goalOrientation)
                    goalHead = Header(0, rospy.Time.now(), "map")
                    goal = PoseStamped(header = goalHead, pose = pose)
                    path = request(start, goal, 0)
                    if len(path.plan.poses) == 0:
                        available = False
                    else:
                        loc_angle.append(path.plan.poses[-1].pose)
                        cv2.rectangle(map,(int(w),height-int(h)),(int(w),height-int(h)),(100,100,100),2)
                    angle  += 1
                    if available:
                        print "planned (x: {}, y: {})".format(w*resolution+origin[0], h*resolution+origin[1])
                if len(loc_angle) > 0:
                    availableLoc.append(loc_angle)
                h+=4 #0.25 m
            w+=4
        # cv2.imshow('map',map)
        cv2.imwrite("planned_map.jpg", map);
        f = open('map_positions', 'wb')
        cPickle.dump(availableLoc, f)
        f.close()
        print "Writing Map Position Finished"

    print "Making folders"
    for idx, plan in enumerate(availableLoc):
        try:
            dirname = "sudo_data_set/{}".format(idx)
            os.makedirs(dirname)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
    print "Making Folders is done"
    
    
    
# #-------------------------------------------------------------------------


    yaml_list = []
    image_list = []
    cvBridge = CvBridge()
    error = 0.001



    print "start dataset creation"
    rospy.wait_for_service('gazebo/set_model_state')
    setModelState = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)

    rospy.wait_for_service('gazebo/pause_physics')
    pause_physics = rospy.ServiceProxy('gazebo/pause_physics', Empty)

    rospy.wait_for_service('gazebo/unpause_physics')
    unpause_physics = rospy.ServiceProxy('gazebo/unpause_physics', Empty)

    print "got service 'gazebo/set_model_state'"
    
    for idx, location in enumerate(tqdm(availableLoc)):
        for plan in location:
            pose = plan
            time_start = time.time()

            state = ModelState(model_name="sudo-spawn", pose=pose, reference_frame="map")
            pause_physics()
            response = setModelState(state)
            time.sleep(0.1)
            unpause_physics()
            time.sleep(0.2)
            image_data = rospy.wait_for_message("/sudo/bottom_webcam/image_raw", Image)
            image = cvBridge.imgmsg_to_cv2(image_data, "bgr8")
            cv2.imshow("AAAAAARGH", image)
            cv2.waitKey(1)
            orientation = (pose.orientation.x, pose.orientation.y,pose.orientation.z,pose.orientation.w)
            yaw = tf.transformations.euler_from_quaternion(orientation)[2]
            total_angle = math.degrees(yaw)
            image_list.append([image, pose.position, total_angle])
            


            #post-processing
        print "write image stack {}".format(idx)
        for image, position, angle in image_list:
            image_location = "sudo_data_set/{}/{}_{:04}.jpg".format(idx, idx, int(angle*10))
            if not os.path.exists(image_location):
                cv2.imwrite(image_location, image);
                yaml_list.append({'image':{'file':image_location,'location':{'x':position.x, 'y':position.y, 'z':position.z}, 'angle':angle}})
                old_angle = int(angle*10)
        with open('data_set.yaml', 'a') as yaml_file:
            yaml.dump(yaml_list, yaml_file, default_flow_style=False)

        print "clear image_list"
        image_list = []
        yaml_list = []

    rospy.spin()


        




