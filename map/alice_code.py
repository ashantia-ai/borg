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

from nav_msgs.msg import Path, Odometry
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





if __name__ ==  '__main__' :
	rospy.init_node("planner")

	yaml_list = []
	image_list = []
	cvBridge = CvBridge()
	error = 0.001


 
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
	idx = 0
	while True:
		face = 0
		while face < 360:
			
			while True:
				odom = rospy.wait_for_message("/odom", Odometry)
				orientation = Quaternion(odom.orientation.x, odom.orientation.y, odom.orientation.z, odom.orientation.w)
				orientation_euler = tf.transformations.euler_from_quaternion(orientation)
			    print "direction face: {} degrees".format(orientation_euler[2])
				i = input('Turn Alice and press enter to continue: ')
			    if not i:
			        break

			panPosition.publish(-pan_pos*math.pi/2)
			print "turning..."
			pan_status = rospy.wait_for_message("/alice/pan_controller/state", JointControllerState)
			while abs(pan_status.error) < 1:
				pan_status = rospy.wait_for_message("/alice/pan_controller/state", JointControllerState)
			while abs(pan_status.error) > error:
				pan_status = rospy.wait_for_message("/alice/pan_controller/state", JointControllerState)
				image_data = rospy.wait_for_message("/front_xtion/rgb/image_raw", Image)
				image = cvBridge.imgmsg_to_cv2(image_data, "bgr8")
				total_angle = (orientation_euler[2]+math.degrees(pan_status.process_value)+360)%360
				image_list.append([image, odom.position, total_angle])
			pan_pos *= -1 

			face += 180

		try:
			dirname = "data_set_real/{}".format(idx)
			os.makedirs(dirname)
		except OSError as exception:
			if exception.errno != errno.EEXIST:
				raise

		#post-processing
		print "write image stack {}".format(idx)
		for image, position, angle in image_list:
			image_location = "data_set_real/{}/{}_{:04}.jpg".format(idx, idx, int(angle*10))
			if not os.path.exists(image_location):
				cv2.imwrite(image_location, image);
				yaml_list.append({'image':{'file':image_location,'location':{'x':position.x, 'y':position.y, 'z':position.z}, 'angle':angle}})
		with open('data_set_real.yaml', 'a') as yaml_file:
			yaml.dump(yaml_list, yaml_file, default_flow_style=False)

		print "clear image_list"
		image_list = []
		yaml_list = []
		idx += 1

	rospy.spin()


		




