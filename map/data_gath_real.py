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
import actionlib

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
from xtationJoint import Joint
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry





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
	startPos = Point(0, 0, 0)
	startOrientation = Quaternion(0,0, 0.05, 1)
	# startPose = Pose()
	startPose = Pose(startPos, startOrientation)
	startHead = Header(0, rospy.Time.now(), "map")
	start = PoseStamped(header = startHead, pose = startPose)

	# w = 0
	# while w < width :
	# 	h = 5
	# 	while h < height :
	# 		angle = int(0)
	# 		available = True
	# 		while angle < 360:
	# 			quat=tf.transformations.quaternion_from_euler(0, 0, math.radians(angle))
	# 			goalPos = Point(w*resolution+origin[0], h*resolution+origin[1], 0)
	# 			# print "checking {}".format(goalPos)
	# 			goalOrientation = Quaternion(quat[0], quat[1], quat[2], quat[3])
	# 			pose = Pose(position = goalPos, orientation = goalOrientation)
	# 			goalHead = Header(0, rospy.Time.now(), "map")
	# 			goal = PoseStamped(header = goalHead, pose = pose)
	# 			path = request(start, goal, 0)
	# 			if len(path.plan.poses) == 0:
	# 				available = False
	# 			angle  += 90
	# 		if available == True:
	# 			print "planned (x: {}, y: {})".format(w*resolution+origin[0], h*resolution+origin[1])
	# 			availableLoc.append(path.plan)
	# 			cv2.rectangle(map,(int(w),height-int(h)),(int(w),height-int(h)),(100,100,100),2)
	# 		h+=5 #0.25 m
	# 	w+=5
	# # cv2.imshow('map',map)
	# cv2.imwrite("planned_map.jpg", map);

	# numpy.save(os.path.join('/home/borg/sudoRepo/map/', 'plannedMap'), availableLoc)
	availableLoc = numpy.load('/home/borg/sudoRepo/map/plannedMap.npy')
	availableLoc = availableLoc[500:600]

# #-------------------------------------------------------------------------
	for idx, plan in enumerate(availableLoc):
		try:
			dirname = "data_set/{}".format(idx)
			os.makedirs(dirname)
		except OSError as exception:
			if exception.errno != errno.EEXIST:
				raise

	yaml_list = []
	image_list = []
	cvBridge = CvBridge()
	error = 0.001



	print "start dataset creation"
	# rospy.wait_for_service('gazebo/set_model_state')
	# setModelState = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)

	# rospy.wait_for_service('gazebo/pause_physics')
	# pause_physics = rospy.ServiceProxy('gazebo/pause_physics', Empty)

	# rospy.wait_for_service('gazebo/unpause_physics')
	# unpause_physics = rospy.ServiceProxy('gazebo/unpause_physics', Empty)

	# print "got service 'gazebo/set_model_state'"

	client = actionlib.SimpleActionClient("move_base", MoveBaseAction)

	tiltPosition = rospy.Publisher("/alice/tilt_controller/command", Float64, queue_size = 1)
	panPosition = rospy.Publisher("/alice/pan_controller/command", Float64, queue_size = 1)

	rospy.sleep(2)

	tiltPosition.publish(math.pi/2)
	panPosition.publish(math.pi/2)

	pan_status = rospy.wait_for_message("/alice/pan_controller/state", JointControllerState)
	tilt_status = rospy.wait_for_message("/alice/tilt_controller/state", JointControllerState)


	while abs(pan_status.error) > error or abs(tilt_status.error) > error:
		pan_status = rospy.wait_for_message("/alice/pan_controller/state", JointControllerState)
		tilt_status = rospy.wait_for_message("/alice/tilt_controller/state", JointControllerState)
	pan_pos = 1

	

	for idx, plan in enumerate(availableLoc):
		pose = plan.poses[-1].pose
		goal = MoveBaseGoal()
		goal.target_pose.pose = pose
		goal.target_pose.header.frame_id = 'map'


		twist = Twist(linear=Vector3(x=0,y=0,z=0), angular=Vector3(x=0,y=0,z=0))
		face = 0
		time_start = time.time()
		while face < 360:
			quat=tf.transformations.quaternion_from_euler(0, 0, math.radians(face))
			pose.orientation = Quaternion(quat[0], quat[1], quat[2], quat[3])
			# state = ModelState(model_name="alice", pose=pose, reference_frame="map") this needs to be replaced by a move command
			# response = setModelState(state)
			goal.target_pose.header.stamp = rospy.Time.now()
			client.send_goal(goal)
			client.wait_for_result()

			if True:  #change to wait for goal or something like that
				odom = rospy.wait_for_message("/odom", Odometry)
				# orientation = Quaternion(odom.orientation.x, odom.orientation.y, odom.orientation.z, odom.orientation.w)
				# orientation_euler = tf.transformations.euler_from_quaternion(orientation)
			    # print "direction face: {} degrees".format(orientation_euler[2])

				angle = pan_pos*math.pi
				panPosition.publish(-pan_pos*math.pi/2)
				pan_status = rospy.wait_for_message("/alice/pan_controller/state", JointControllerState)
				while abs(pan_status.error) < 1:
					pan_status = rospy.wait_for_message("/alice/pan_controller/state", JointControllerState)
				while abs(pan_status.error) > error:
					pan_status = rospy.wait_for_message("/alice/pan_controller/state", JointControllerState)
					image_data = rospy.wait_for_message("/front_xtion/rgb/image_raw", Image)
					image = cvBridge.imgmsg_to_cv2(image_data, "bgr8")
					total_angle = (face+math.degrees(pan_status.process_value)+360)%360
					image_list.append([image, odom.pose.pose, total_angle])
				pan_pos *= -1 
			face += 180
		expected_time = (time.time() - time_start) * (len(availableLoc)-idx)
		m, s =  divmod(expected_time , 60)
		h, m = divmod(m, 60)
		print "expected time remaining {}:{}:{}".format(h, m, s)


		# #post-processing
		# print "write image stack {}".format(idx)
		# for image, position, angle in image_list:
		# 	image_location = "data_set_real/{}/{}_{:04}.jpg".format(idx, idx, int(angle*10))
		# 	if not os.path.exists(image_location):
		# 		cv2.imwrite(image_location, image);
		# 		yaml_list.append({'image':{'file':image_location,'location':{'x':position.x, 'y':position.y, 'z':position.z}, 'angle':angle}})
		# 		old_angle = int(angle*10)
		# with open('data_set.yaml', 'a') as yaml_file:
		# 	yaml.dump(yaml_list, yaml_file, default_flow_style=False)

		# print "clear image_list"
		# image_list = []
		# yaml_list = []

	rospy.spin()


		




