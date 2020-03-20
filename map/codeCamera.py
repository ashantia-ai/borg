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

from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from xtationJoint import Joint
from control_msgs.msg import JointControllerState
from std_msgs.msg import Float64
from std_srvs.srv import Empty




if __name__ ==  '__main__' :
	rospy.init_node("planner")


	availableLoc = []
	mapVar = yaml.load(open("sim_map.yaml", 'r'))
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
	startPos = Point(0, 0, 0.25)
	startOrientation = Quaternion(0,0, 0.05, 1)
	# startPose = Pose()
	startPose = Pose(startPos, startOrientation)
	startHead = Header(0, rospy.Time.now(), "map")
	start = PoseStamped(header = startHead, pose = startPose)

	w = 0
	while w < width :
		h = 5
		while h < height :
			angle = int(0)
			available = True
			while angle < 360:
				quat=tf.transformations.quaternion_from_euler(0, 0, math.radians(angle))
				goalPos = Point(w*resolution+origin[0], h*resolution+origin[1], 0.19)
				# print "checking {}".format(goalPos)
				goalOrientation = Quaternion(quat[0], quat[1], quat[2], quat[3])
				pose = Pose(position = goalPos, orientation = goalOrientation)
				goalHead = Header(0, rospy.Time.now(), "map")
				goal = PoseStamped(header = goalHead, pose = pose)
				path = request(start, goal, 0)
				if len(path.plan.poses) == 0:
					available = False
				angle  += 90
			if available == True:
				print "planned (x: {}, y: {})".format(w*resolution+origin[0], h*resolution+origin[1])
				availableLoc.append(path.plan)
				cv2.rectangle(map,(int(w),height-int(h)),(int(w),height-int(h)),(100,100,100),2)
			h+=5 #0.25 m
		w+=5
	# cv2.imshow('map',map)
	cv2.imwrite("planned_map.jpg", map);


# #-------------------------------------------------------------------------
	for idx, plan in enumerate(availableLoc):
		try:
			dirname = "data_set/{}".format(idx)
			os.makedirs(dirname)
		except OSError as exception:
			if exception.errno != errno.EEXIST:
				raise

	delta_angle = 5
	tilt_nav = 1.57
	yaml_list = []
	image_list = []
	cvBridge = CvBridge()


	print "start dataset creation"
	rospy.wait_for_service('gazebo/set_model_state')
	try:
		setModelState = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e

	try:
		getModelState = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e

	try:
		pause_physics = rospy.ServiceProxy('gazebo/pause_physics', Empty)
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e

	try:
		unpause_physics = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e

	tiltPosition = rospy.Publisher("alice/tilt_controller/command", Float64, queue_size = 1)
	panPosition = rospy.Publisher("alice/pan_controller/command", Float64, queue_size = 1)


	print "got service 'gazebo/set_model_state'"
	# #TODO: teleport to goal, rotate and take pictures
	for idx, plan in enumerate(availableLoc):
		pose = plan.poses[-1].pose
		twist = Twist(linear=Vector3(x=0,y=0,z=0), angular=Vector3(x=0,y=0,z=0))
		angle = int(0)
		positioning = True
		#while angle < 360:
		pause_physics()
		quat=tf.transformations.quaternion_from_euler(0, 0, math.radians(angle))
		pose.orientation = Quaternion(quat[0], quat[1], quat[2], quat[3])
		state = ModelState(model_name="alice", pose=pose, reference_frame="map")
		response = setModelState(state)
			#aliceLoc = getModelState("alice", "map")
		time.sleep(0.5)
		unpause_physics()
		while positioning:
			print "start positioning"
			panPosition.publish(-1.57) # publish the pan position
			panValue = rospy.wait_for_message("/alice/pan_controller/state", JointControllerState)
			if (panValue.process_value <-1.55 and panValue>-1.60):
				positioning = False


		print "positioned"
		while panValue.process_value<1.55:
			print "start gathering"
			panPosition.publish(1.57)
			panValue = rospy.wait_for_message("/alice/pan_controller/state", JointControllerState)
			image_data = rospy.wait_for_message("/front_xtion/rgb/image_raw", Image)
			angle = panValue.process_value*180/3.14
			image = cvBridge.imgmsg_to_cv2(image_data, "bgr8")
			image_list.append([image, pose.position, angle])
		time.sleep(0.5)
			#if (response.success == True):
		
				#time.sleep(0.1)
				#image_data = rospy.wait_for_message("/front_xtion/rgb/image_raw", Image)
				#image = cvBridge.imgmsg_to_cv2(image_data, "bgr8")
				#image_list.append([image, pose.position, angle])
				#angle += delta_angle

		print "write image stack {}".format(idx)
		for image, position, angle in image_list:
			image_location = "data_set/{}/{}_{}.jpg".format(idx, idx, int(angle))
			cv2.imwrite(image_location, image);
			yaml_list.append({'image':{'file':image_location,'location':{'x':position.x, 'y':position.y, 'z':position.z}, 'angle':angle}})
		image_list[:] = []
		print "clear image_list"


		

	with open('data_set.yaml', 'w') as yaml_file:
		yaml.dump(yaml_list, yaml_file, default_flow_style=False)

	image_list[:] = []
	cv2.waitKey(0)
	cv2.destroyAllWindows()




