#!/usr/bin/env python
import matplotlib.pyplot as plt
import rospy, roslib
from rospy import ROSException, ROSTimeMovedBackwardsException

import cPickle, sys
import math, numpy, cv2
import numpy as np
import copy
import os
import dill

import rl_methods.nfq as nfq

import time
import theano
from theano import tensor as T, config, shared
from theano.tensor.shared_randomstreams import RandomStreams

import std_srvs.srv
import stage_ros.srv

import sensor_msgs
import rl_methods.nfq 

from sensor_msgs.msg import Image, CameraInfo, Imu
from cacla_nav.msg import visualize
from std_msgs.msg import Bool, Int8
from geometry_msgs.msg import Twist
from geometry_msgs.msg._Quaternion import Quaternion
from nav_msgs.msg import Odometry

from robot_control.srv import *
from robot_control.msg import *

from move_base_msgs.msg import *

import actionlib
import actionlib_msgs.msg

import tf

from cv_bridge import CvBridge, CvBridgeError

import multiprocessing
from multiprocessing import Lock, Process, Value, Array, Manager

# import figure_make

import os
import Queue
from numpy.f2py.auxfuncs import throw_error

manager = Manager()
NUM1 = Value('i', 1)
ARR1 = manager.list()
LOCK1 = Lock()

NUM2 = Value('i', 1)
ARR2 = manager.list()
LOCK2 = Lock()

simulation_env = "g_big_start.sh"


def normalize(scale):    
    maxX = 7.3
    minY = -1.9
    maxY = 2.57 + -minY
    
    # normalize the x 
    scale[0] /= maxX
    
    # normalize the y
    
    scale[1] += -minY
    scale[1] /= maxY
    
    # normalize the sin and cos by adding 1 and divide by 2
    scale[2] += 1
    scale[2] /= 2
    scale[3] += 1
    scale[3] /= 2
    
    return scale


def scale(scale):    
    maxX = 7.3
    minY = -1.9
    maxY = 2.57 + -minY
    
    scale[0] *= maxX
    scale[1] *= maxY
    scale[1] -= -minY
    return scale

    
class Position(object):

    def __init__(self, x = 0, y = 0, z = 0, roll = 0, pitch = 0, yaw = 0):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw


class CACLA_nav(object):

    def __init__(self, base_path, reference_frame, x_init, y_init, goal = (-3.53, -6.14, 0.0), ordered_list_index = [(0, 0, 0)], **kwargs):
        
        self.time_to_finish_mb_action = 30.0
        # Prepares Action-Service for NFQ
        self.pose_index = 0        
        self.input_type = "ground_truth"  # Options: 1- autoencoder . 2-location, 3- ground truth location, Default: ground truth
        self.method = "nfq"  # 1- Cacla 2- NFQ
        
        self.simul_path = os.environ['BORG'] + '/ros/catkin_ws/src/sudo_stage/launch/'
        print self.simul_path
        
        self.minX = -0.5
        self.minY = -3.75
        self.maxX = 5.0
        self.maxY = 0.55
        
        self.steps_number = 1
        self.trial_max_steps = 1000

        self.ordered_list_index = ordered_list_index
        self.launch_sim_and_world(0)
        
        self.lowest_path = base_path
        self.base_path = os.path.join(base_path, 'goal_' + str(self.ordered_list_index[self.pose_index][0]) + '/')
        self.current_goal = self.ordered_list_index[self.pose_index][0]
        print "Base Path is:" , self.base_path
        
        self.x_init_list = x_init
        self.y_init_list = y_init
        self.goals = goal
        
        self.x_init = self.x_init_list[self.pose_index]
        self.y_init = self.y_init_list[self.pose_index]
        
        self._init_ros_node()
        self.transformer = tf.TransformListener()

        self.__prepare_action_service()
        
        self._prepare_file_structure(self.base_path)
        
        self.reference_frame = reference_frame
        self.transform_listener = tf.TransformListener()
        self.bridge = CvBridge()
        
        # goal = self._convert_from_grid(goal[0], goal[1])
           
        # Map Size in meters
        # TODO:Watch out here, x_len and y_len was 11. It should the max length in meters
        # self.x_len = 11 #self.maxX + math.fabs(self.minX)
        # self.y_len = 14 #self.maxY + math.fabs(self.minY)
        self.x_len = 47  # self.maxX + math.fabs(self.minX)
        self.y_len = 24  # self.maxY + math.fabs(self.minY)
        # self.metric_resolution = 0.5
        self.metric_resolution = 0.4
        self.structure = self.x_len * self.y_len  # * (1 / self.metric_resolution)
        self._prepare_init_n_goal_position((self.goals[self.pose_index][0], self.goals[self.pose_index][1], 0))
        self.__prepare_rewards()
        
        self.__adjust_sensor_multipliers()
        self.__prepare_action_variables()
        
        self.__prepare_performance_variables()
        self.__prepare_experience_replay()
        
        # self.imu_test = numpy.zeros((1,5))
        # self.imu_count = 0
        
        self.steps_data = []
        self.reward_data = []
        self.figure_update_frequency = 1

        self.consecutive_succes = list(np.zeros(15))
        self.actions_numbers = list(np.zeros(200))
        self.fig1_data = []       
        self.fig2_data = []
        
        self.nfq_args = {'epsilon':(0.1, 0.1) \
                , 'discount_factor':0.9, 'random_decay': 0.995\
                , 'explore_strategy':3, 'activation_function':1
                , 'learning_rate':0.3}
        
        if kwargs:
            self.set_parameters(**kwargs)
        
        self._convert_to_grid(self.x, self.y, self.radian)  # Initializes empty vector size
        self._prepare_RL_variables()
        
        self.__action_result = {'state':None, 'done':False, 'res':None}
        
        self.reset()
        
        self._init_subscribers()
        self.save_timer = time.time()
        self.save_interval = 3600.0  # seconds
        self.failed_action_queue = []
        
    def launch_sim_and_world(self, ind):
        global simulation_env
        print 'killing existing worlds'
        os.system('killall xterm -9')
        time.sleep(2)
    
        print 'lunching word ', ind
        os.system('sh ' + self.simul_path + simulation_env + " " + str(ind))
        time.sleep(2)
        print 'world created'
    
    def __repr__(self, *args, **kwargs):
        return object.__repr__(self, *args, **kwargs)
    
    def __adjust_sensor_multipliers(self):
        if self.input_type == "autoencoder": 
            self.bumper_value = 5  # This value will be sent to NN, it is higher than one to show importance
            self.imu_multiplier = 5  # This value will be multiplied by imu values, it is higher than one to show importance
            self.speed_multiplier = 5  # This value will be multiplied by speed values, it is higher than one to show importance
        else:
            self.bumper_value = self.imu_multiplier = self.speed_multiplier = 1  # There are only location values, so no multiplying
        
        self.fbumper = self.bbumper = self.lbumper = self.rbumber = self.chassis = -self.bumper_value
        self.in_collision = False
        self.linear_speed = self.angular_speed = 0
        self.imu = numpy.zeros((1, 5))
        
    def _convert_from_grid(self, x, y):

        # precise_x = x * self.metric_resolution + self.minX + self.metric_resolution / 2
        # precise_y = y * self.metric_resolution + self.minY + self.metric_resolution / 2
        
        step_size_y = step_size_x = 0.4
        
        precise_y = (-x * step_size_x) - 0.2
        precise_x = (y * step_size_y) + 0.2
        
        # y2=(-x*step_size_x) - 0.25 + start_y
        
        return precise_x, precise_y 

    def _convert_to_grid(self, x, y, theta):
        '''
        Converts the location output to grid like format
        '''
       
        # y += -self.minY
        # x += -self.minX
    
        step_size_y = step_size_x = 0.4
        
        columns = self.x_len  # int(math.ceil(self.x_len / self.metric_resolution))
        rows = self.y_len  # int(math.ceil(self.y_len / self.metric_resolution))
                
        # x_idx = int(x / self.metric_resolution)
        # y_idx = int(y / self.metric_resolution)
        
        x_idx = (y) / -step_size_x
        
        print "x_idx:", x_idx
        x_idx = int(x_idx)
        y_idx = (x) / step_size_y
        print "y_idx", y_idx
        y_idx = int(y_idx)
        
        # start_y = 0.65
        # start_x = -0.5
        # x2=(y*step_size_y) + 0.25 + start_x
        # y2=(-x*step_size_x) - 0.25 + start_y
        
        pos_matrix = numpy.zeros((columns, rows), dtype = theano.config.floatX)
        try:
            pos_matrix[int(x_idx), int(y_idx)] = 1
        except:
            print x, ":", y, ":", theta
            print x_idx, ":", y_idx
            raise
        
        self.empty_vector = numpy.zeros((1, pos_matrix.size))
        self.empty_vector = numpy.reshape(self.empty_vector, -1)
        
        return numpy.reshape(pos_matrix, -1), (x_idx, y_idx)
              
    def _create_directories(self):
        if not os.path.exists(self.base_path):
            raise Exception("Base path %s doesn't exist" % (self.base_path))
        elif os.path.exists(self.method_path):
            try:
                os.mkdir(self.method_path)
            except Exception, e:
                print repr(e)
                
    def _init_ros_node(self):
        ''' Initialized the ros node and the main publishers
        '''
        rospy.loginfo("Callbacks registered")
        try:
            rospy.init_node('Simple_nav')
        except Exception as e:
            print e
        rospy.loginfo("Node initialized")
        
        self.cacla_pub = rospy.Publisher('cacla_actions', visualize, queue_size = 1)
        self.action_pub = rospy.Publisher('rl_actions', Int8, queue_size = 1)
        self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size = 1)
        
        self.client = actionlib.SimpleActionClient('readaction', robot_control.msg.rlAction)
        self.movebase_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        # self.client.wait_for_server()
        self.movebase_client.wait_for_server()
        rospy.loginfo("Move_base client connected.")
        # rospy.wait_for_service('reset_positions')
        rospy.wait_for_service('reset_positions_with_pose', timeout = 10)
        self.reset_stage = rospy.ServiceProxy('reset_positions_with_pose', stage_ros.srv.resetToPose)
        pose = stage_ros.srv.resetToPoseRequest()
        pose.x = self.x_init
        pose.y = self.y_init
        self.reset_stage(pose)
        print "Stage ROS is ready..."
        
        # Frequency of the Imagecb function
        self.loop_rate = rospy.Rate(100)
        # Used to make sure robot will stop before reseting the trial
        self.rate = rospy.Rate(100)
        
    def _init_subscribers(self):
        self.col_subscriber = rospy.Subscriber("/collision", Bool, self.chassiscb, queue_size = 1)
        # self.odom_subscriber = rospy.Subscriber("/base_pose_ground_truth", Odometry, self.odomcb, queue_size=1)
    
    def __del__(self):
        print "destructor called"
        
    def __load_network(self):
        try:
            f = file(self.theano_file, 'rb')
            model = cPickle.load(f)
            f.close()
            return model
        except Exception, e:
            raise Exception("Error in loading network: %s " % (repr(e))) 
    
    def __perform_pedal_action(self, action):

        def scale_action(action):
            pass
        
        linear_acc, angular_acc = action
        
        linear_speed = self.last_linear_speed + linear_acc * self.linear_acc_step
        if linear_speed > 0:
            linear_speed = min(linear_speed, self.maximum_linear_speed)
        else:
            linear_speed = max(linear_speed, -self.maximum_linear_speed)
            
        angular_speed = self.last_angular_speed + angular_acc * self.angular_acc_step
        if angular_speed > 0:
            angular_speed = min(angular_speed, self.maximum_angular_speed)
        else:
            angular_speed = max(angular_speed, -self.maximum_angular_speed)
        
        # print "Linear Speed: ", linear_speed, " Angular Speed: ", angular_speed
        # create a twist message, fill in the details
        twist = Twist()
        twist.linear.x = linear_speed  # our forward speed
        twist.linear.y = 0; twist.linear.z = 0;  # we can't use these!        
        twist.angular.x = 0; twist.angular.y = 0;  #          or these!
        twist.angular.z = angular_speed;  # no rotation
        
        self.publisher.publish(twist)
    
    def __perform_action(self, action):

        def scale_action(action):
            pass

        # TODO: add max , min speed limitation. BUT NOT HERE, in CACLA 
        linear_speed, angular_speed = action
        
        # print "Linear Speed: ", linear_speed, " Angular Speed: ", angular_speed
        # create a twist message, fill in the details
        twist = Twist()
        twist.linear.x = linear_speed  # our forward speed
        twist.linear.y = 0; twist.linear.z = 0;  # we can't use these!        
        twist.angular.x = 0; twist.angular.y = 0;  #          or these!
        twist.angular.z = angular_speed;  # no rotation
        
        self.publisher.publish(twist)
        
    def _perform_movebase_action(self, action_num, state):
        
        test = rospy.Time.now()
        goal = MoveBaseGoal()
        
        x, y, theta = state
        theta = self.radian
        epsilon = 0.1
        
        log = "\n    Current Pos: X: %s, Y: %s , Theta: %s" % (x, y, theta)
        if action_num == 0:
            if theta >= -math.pi / 4 and theta < math.pi / 4:
                x += 1
            elif theta >= math.pi / 4 and theta < math.pi * 3 / 4:
                y += 1
            elif theta >= math.pi * 3 / 4 or theta < -math.pi * 3 / 4:
                x -= 1
            elif theta >= -math.pi * 3 / 4 and theta < -math.pi / 4:
                y -= 1
            else:
                raise
        
        elif action_num == 1:
            if theta >= -math.pi / 4 and theta < math.pi / 4:
                x -= 1
            elif theta >= math.pi / 4 and theta < math.pi * 3 / 4:
                y -= 1
            elif theta >= math.pi * 3 / 4 or theta < -math.pi * 3 / 4:
                x += 1
            elif theta >= -math.pi * 3 / 4 and theta < -math.pi / 4:
                y += 1
            else:
                raise
            
        elif action_num == 2:
            theta += math.pi / 2
            if theta >= math.pi:
                theta -= 2 * math.pi
            
        elif action_num == 3:
            theta -= math.pi / 2
            if theta < -math.pi:
                theta += 2 * math.pi
                
        elif action_num == 4:
            return 0
        
        # x_check = lambda f: f < 0 or f >= self.structure.shape[0]
        # y_check = lambda f: f < 0 or f >= self.structure.shape[1] 
        # if x_check(x) or y_check(y) or self.structure[x, y] == 1:
        #    self.in_collision = True
        #    rospy.logdebug("Destination Location is X: %s, Y: %s, T: %s COLLISION with Action %s" % (x, y, theta, action_num))
        #    return False
        
        log += "\nDestination Pos: X: %s, Y: %s , Theta: %s" % (x, y, theta)
        # rospy.loginfo(log)
        x, y, theta = self._convert_from_grid(x, y, theta)
        
        rospy.logdebug("Real Destination Location is X: %s, Y: %s, T: %s, action num: %s" % (x, y, theta, action_num))
        # print "x: %s, y: %s, theta: %s" % (x,y,theta)
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]
        goal.target_pose.header.frame_id = 'odom'
        goal.target_pose.header.stamp = rospy.Time.now()
        
        results = self.movebase_client.send_goal(goal, done_cb = self.__movebase_done_cb)
        
        action_timer = rospy.Time.now()
        while not self.__action_result['done']:
             
            if rospy.Time.now().to_sec() - action_timer.to_sec() > 10.0:
                self.in_collision = True
                self.movebase_client.cancel_all_goals()
                self.movebase_client.cancel_goal()
                rospy.logwarn("Action didn't finish in time")
                return
            
        self.__action_result['done'] = False 
        if self.__action_result['state'] != actionlib_msgs.msg.GoalStatus.SUCCEEDED:
            f_reason = self.__action_result['state']
            reason = "Other reasons."
            if f_reason == actionlib_msgs.msg.GoalStatus.PREEMPTED:
                reason = "Goal Cancel was requested but operation completed"
            elif f_reason == actionlib_msgs.msg.GoalStatus.ABORTED:
                reason = "Goal Aborted."
            elif f_reason == actionlib_msgs.msg.GoalStatus.LOST:
                reason = "Goal was LOST. WARNING. SOMETHING IS WRONG."
            elif f_reason == actionlib_msgs.msg.GoalStatus.REJECTED:
                reason = "Goal was REJECTED"
            
            rospy.logwarn("Action failed with Reason: " + str(reason))
            self.in_collision = True
            self.movebase_client.cancel_all_goals()
            self.movebase_client.cancel_goal()
            # rospy.sleep(rospy.Duration(1))
        
        return
        
    def _perform_movebase_action_no_rot(self, action_num, state):
        test = rospy.Time.now()
        goal = MoveBaseGoal()
        x, y = state
        epsilon = 0.1
        yaw = 0
        
        log = "\n    Current Pos: X: %s, Y: %s" % (x, y)
        if action_num == 0:
            x -= 1
            yaw = 1.57
        elif action_num == 1:
            y -= 1
            yaw = 3.139
        elif action_num == 2:
            x += 1
            yaw = -1.57
        elif action_num == 3:
            y += 1
            
        elif action_num == 4:
            pass
        
        log += "\nDestination Pos: X: %s, Y: %s " % (x, y)
        rospy.logdebug(log)
        x, y = self._convert_from_grid(x, y)
        
        rospy.logdebug("Real Destination Location is X: %s, Y: %s, action num: %s" % (x, y, action_num))
        # print "x: %s, y: %s, theta: %s" % (x,y,theta)
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]
        goal.target_pose.header.frame_id = 'odom'
        goal.target_pose.header.stamp = rospy.Time.now()
        
        results = self.movebase_client.send_goal(goal, done_cb = self.__movebase_done_cb)
        self.movebase_client.wait_for_result(rospy.Duration(self.time_to_finish_mb_action))
        
        action_timer = rospy.Time.now()
        while not self.__action_result['done']:
             
            if rospy.Time.now().to_sec() - action_timer.to_sec() > self.time_to_finish_mb_action:
                self.in_collision = True
                self.movebase_client.cancel_all_goals()
                self.movebase_client.cancel_goal()
                rospy.logwarn("Action didn't finish in time")
                self.__action_result['done'] = False
                return False
            
        self.__action_result['done'] = False 
        if self.__action_result['state'] != actionlib_msgs.msg.GoalStatus.SUCCEEDED:
            f_reason = self.__action_result['state']
            reason = "Other reasons."
            if f_reason == actionlib_msgs.msg.GoalStatus.PREEMPTED:
                reason = "Goal Cancel was requested but operation completed"
                return True
            elif f_reason == actionlib_msgs.msg.GoalStatus.ABORTED:
                reason = "Goal Aborted."
            elif f_reason == actionlib_msgs.msg.GoalStatus.LOST:
                reason = "Goal was LOST. WARNING. SOMETHING IS WRONG."
            elif f_reason == actionlib_msgs.msg.GoalStatus.REJECTED:
                reason = "Goal was REJECTED"
            
            rospy.logwarn("Action failed with Reason: " + str(reason))
            self.in_collision = True
            return False
            # self.movebase_client.cancel_all_goals()
            # self.movebase_client.cancel_goal()
            # rospy.sleep(rospy.Duration(1))

        return True

    def __movebase_done_cb(self, terminal_state, result):
        self.__action_result = {'state':terminal_state, 'done':True, 'res':result}
    
    def __perform_discreet_action(self, action_num):
        
        test = rospy.Time.now()
        goal = robot_control.msg.rlGoal()
        goal.action = action_num
        
        self.client.send_goal(goal)
        
        finished_before_timeout = self.client.wait_for_result(rospy.Duration.from_sec(10))
        if finished_before_timeout:
            rospy.logdebug("Action Time: %s" % (rospy.Time.now().to_sec() - test.to_sec()))
        else:
            rospy.logwarn("Action didn't finish")
        
        return self.client.get_result()
        # print "result is:", response.result

        # TODO:change to service
    def __prepare_action_service(self):
        pass
    
    def __prepare_action_variables(self):
        # TODO: needs to called from set_params too
        # Action Variables
        self.maximum_linear_speed = 0.4  # m/s
        self.maximum_angular_speed = 1.57  # rad/s
        self.linear_acc_step = 0.04
        self.angular_acc_step = 1
        self.last_linear_speed = 0
        self.last_angular_speed = 0
        if self.method == "cacla":
            self.previous_action = [0, 0]
        else:
            self.previous_action = 0

    def __prepare_experience_replay(self):
        self.state_history = []
        self.success_history = []
        self.replay_frequency = 100
        self.train_frequency = 10
        self.train_frequency_action = 100
        self.train_frequency_step = 100
        self.replay_frequency_action = 500
        self.replay_frequency_step = 500
               
    def _prepare_file_structure(self, base_path):
        self.base_path = base_path
        self.method_path = os.path.join(self.base_path, self.method)
        print "method path", self.method_path
        self._create_directories()
        
    def _prepare_init_n_goal_position(self, goal = (-3.53, -6.14, 0.0),):
        # Name of the model to control through code
        self.gazebo_model = "sudo-spawn"
        
        self.sudo_initial_position = Position(self.x_init, self.y_init, 0.2, 0, 0, math.radians(0))  # next to curtain side counter, facing down
        # self.sudo_initial_position = Position(2.9, 0.04, 0.02, 0, 0, -1.57) #next to curtain side counter, facing left
        # self.sudo_initial_position = Position(2.53, 0.52, 0.02, 0, 0, -1.57) #next to curtain side counter, by the choke hole, ficing left
        # self.sudo_initial_position = Position(4.8, -0.852, 0.02, 0, 0, 3.14) #next to fridge, facing up
        # self.sudo_initial_position = Position(4, -0.5, 0.02, 0,0,math.radians(0)) #Right on goal
        
        self.x = self.sudo_initial_position.x
        self.y = self.sudo_initial_position.y
        self.sin_theta = math.sin(self.sudo_initial_position.yaw)
        self.cos_theta = math.cos(self.sudo_initial_position.yaw)
        self.degrees = math.degrees(self.sudo_initial_position.yaw)
        self.radian = self.sudo_initial_position.yaw
        # Initial Position update after a successfull learning
        self.yaw_change = 90  # in degrees
        self.x_change = 50  # in cm
        self.y_change = 50  # in cmF
        
        # Goal Parameters
        self.goal_list = []
        self.goal_theta_threshold = 360  # Degrees
        self.goal_theta_linear_threshold = 90  # Degrees
        self.goal_xy_threshold = 0.4  # Meters
        self.goal_xy_linear_threshold = 0.5  # Meters
        
        # self.goal = {'x':3.66, 'y':-0.59, 'theta':-90} #Next to stove, facing to it
        self.goal = {'x':goal[0], 'y':goal[1], 'theta':goal[2]}  # infront of init position.
        print "goal when created is: ", self.goal
        
    def _prepare_next_goal(self):
        ''' Reset all RL data, and goes to the next goal.
        Exits if all goals are reached.
        '''
        rospy.loginfo('learned This Goal, Saving Analysis and changing active goal')
        
        self.pose_index += 1
        
        if self.pose_index >= len(self.goals):
            print "Signaling Shutdown"
            self.save_state(force = True)
            raise rospy.ROSInterruptException("Shutdown Signalled. Breaking out of program")
           
        self.x_init = self.ordered_list_index[self.pose_index][1]
        self.y_init = self.ordered_list_index[self.pose_index][2]
        self.base_path = os.path.join(self.lowest_path, 'goal_' + str(self.ordered_list_index[self.pose_index][0]) + '/')
        
        self._prepare_init_n_goal_position((self.goals[self.pose_index][0], self.goals[self.pose_index][1], 0))
        
        # TODO: detect whether current goal changed.
        if self.ordered_list_index[self.pose_index][0] != self.current_goal:
            rospy.loginfo("Goal Changed. Reinitializing RL parameters")
            self.current_goal = self.ordered_list_index[self.pose_index][0]
            self._prepare_file_structure(self.base_path)
            self._prepare_RL_variables()
        self.__reset_performance_variables()
        self.consecutive_succes = list(numpy.zeros(15))
    
    def __prepare_performance_variables(self):
        
        self.queue_size = 10
        self.total_reward_queue = Queue.Queue(self.queue_size)
        
        self.acceptable_run_threshold = 50  # Average total reward of 100 runs should be bigger than 50
        self.success_trials_threshold = 10
        self.success_ratio_threshold = 0.8  # The winning ratio required for updating the initial position
        self.success_trials = 0.0 
        self.failed_trials = 0.0
        self.last_position_update = 0
        self.fail_trial_threshold = 3000  # Number of failed trials before sigma is resetted to a higher value
        self.min_sigma_update = 0.3  # Only reset sigma if it is smaller than this value
        self.win_ratio = 0
        
    def __prepare_rewards(self):
        # The reward if location of the robot is close to goal
        self.location_fix_reward = 0.5
        # The reward if also the angle is correct
        self.angular_fix_reward = 100
        
        # Negative reward per time step
        self.fix_punishment = -0.1
        # Punishment for not reaching the goal
        self.trial_fail_punishment = -0.1
        # Punishmend for hitting an obstacle
        self.obstacle_punishment = -2.0
        self.negative_speed_punishment = -0.0
        
        self.ll_reward = lambda i:-8 * i + 9  # linear location formula - 1 meter is limit
        self.ld_reward = lambda i:-0.2 * i + 7  # linear angular formula - 30 degrees is limit
        self.complex_reward = False  # Enables/Disables smooth reward reduction aroud the goal
        self.stop_at_goal = False
        
    def _prepare_RL_variables(self):
        action_outputs = 2
        nfq_action_outputs = 4
        
        if self.method == ("cacla"): 
            
            self.RL = util.cacla.CACLA(self.method_path, action_outputs, **self.cacla_args)
        elif self.method == ("nfq"):
            self.RL = rl_methods.nfq.NFQ(self.method_path, nfq_action_outputs, self.empty_vector.size, **self.nfq_args)
        else:
            self.RL = rl_methods.nfq.NFQ(self.method_path, nfq_action_outputs, **self.nfq_args)
        
        self.trial_length = 1000  # seconds
        self.trial_begin = rospy.Time().now().to_time()
        self.trial_time = lambda : rospy.Time().now().to_time() - self.trial_begin 
        self.running = True
        
    def __reset_performance_variables(self):
        self.total_reward_queue = Queue.Queue(self.queue_size)
        self.success_ratio_threshold = 0.8  # The winning ratio required for updating the initial position
        self.success_trials = 0.0 
        self.failed_trials = 0.0
        self.last_position_update = 0
        self.win_ratio = 0    
        
    def __visualize__(self, action, explore):
        actions = visualize()
        
        actions.action.linear.x = action[0]
        actions.action.angular.z = action[1]
        
        actions.explore_action.linear.x = explore[0]
        actions.explore_action.angular.z = explore[1]
        
        self.cacla_pub.publish(actions)
                
    def analyze_experiment(self):
        rewards = []
        f = open(os.path.join(self.method_path, "analysis.txt"), 'a')
        while True:
            try:
                rewards.append(self.total_reward_queue.get_nowait())
            except Queue.Empty:
                average_run = numpy.mean(numpy.asarray(rewards))
                report = self.get_current_status(average_run)
                
                self.win_ratio = float(self.success_trials) / float(self.success_trials + self.failed_trials)
                print "SUCESS RUNS FOR THIS ANALYSIS IS %f" % (self.win_ratio)
                report += "Current Win Ration: %f \n" % (self.win_ratio)
                if  self.win_ratio >= self.success_ratio_threshold:
                    self.last_position_update = self.RL.progress
                    self.success_trials = 0
                    self.failed_trials = 0
                    report += "Learning done. Updating the initial Position\n"
                    report += "--------------------------------------------\n"
                    # TODO: perhaps this can be used to update init positions?
                    # self.update_initial_position(0,0,self.yaw_change)
                     
                break
        f.write(report)
        f.close()

    def get_current_status(self, score = 0):
        status = "------ Iteration No %s ------\n" % (self.RL.progress)
        status += "Start Position X: %s Y: %s Theta: %s\n" % (self.sudo_initial_position.x,
                                                              self.sudo_initial_position.y, math.degrees(self.sudo_initial_position.yaw) % 360)
        status += "Goal Position X: %s Y: %s Theta: %s\n" % (self.goal['x'], self.goal['y'], self.goal['theta'])
        status += "Average reward over last %s runs: %s\n\n" % (self.queue_size, score)
        status += "Changing Parameters Section:\n"
        status += "    sigma: %f\n" % (self.RL.epsilon)
        
        return status
                
    def update_initial_position(self, d_x = 0, d_y = 0, d_yaw = 0):
        self.sudo_initial_position.x += d_x
        self.sudo_initial_position.y += d_y
        self.sudo_initial_position.yaw = self.sudo_initial_position.yaw + math.radians(d_yaw)  
           
    def set_parameters(self, **kwargs):
        """
        Set the parameters for the experiment. Each parameter may be specified as
        a keyword argument. Available parameters are:
        """
        for key, value in kwargs.iteritems():
            if key == "goal_position":
                self.goal_position = {'x':float(value[0]), 'y':float(value[1]), 'theta':float(value[2])}
            elif key == "initial_position":
                value = [float(x) for x in value]
                self.sudo_initial_position = Position(*tuple(value))
            elif key == "goal_list":
               self.goal_list = value
            elif key == "goal_theta_threshold":
                self.goal_theta_threshold = int(value)
            elif key == "goal_xy_threshold":
                self.goal_xy_threshold = int(value)
            elif key == "input_type":
                self.input_type = value
                if self.input_type == "autoencoder":
                    self.__adjust_sensor_multipliers()
            elif key == "cacla_args":
                self.cacla_args = value
            elif key == "loop_rate":
                self.loop_rate = rospy.Rate(value)
            elif key in self.__dict__:
                self.__dict__[key] = float(value)
            elif not key in self.__dict__:
                raise Exception("Unknown setting: %s = %s" % (key, repr(value)))
            
    def reset(self):
    
        sanity = False
        s_counter = 0 
        while not sanity:
    
            self.movebase_client.cancel_goals_at_and_before_time(rospy.Time.now())
            self.movebase_client.cancel_all_goals()
            
            # self.__launch_sim_and_world(self.world_ind)
            # self.transformer.clear()
            # self.movebase_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
            # self.client.wait_for_server()
            # self.movebase_client.wait_for_server()
            # rospy.loginfo("Move_base client connected.")
            print 'resetting stage with pose...'
            # TODO: IS THIS CORRECT? should x and y be flipped
            try:
                #rospy.wait_for_service('reset_positions_with_pose', timeout = 10)
                #self.reset_stage = rospy.ServiceProxy('reset_positions_with_pose', stage_ros.srv.resetToPose)
                pose = stage_ros.srv.resetToPoseRequest()
                pose.x = self.x_init
                pose.y = self.y_init
                #print pose.x
                #print pose.y
                self.reset_stage(pose)
                sanity = True
            except ROSException as e:
                rospy.logerr("Reseting pose failed. Something is Wrong.")
                print e
                rospy.sleep(2)
                continue
            except Exception as e:
                print repr(e)
                exit()        
        
        rospy.sleep(1)
        goal = MoveBaseGoal()
        goal.target_pose.pose.position.x = self.x_init
        goal.target_pose.pose.position.y = self.y_init
        quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]
        goal.target_pose.header.frame_id = 'odom'
        goal.target_pose.header.stamp = rospy.Time.now()
        self.movebase_client.send_goal(goal)
        self.movebase_client.wait_for_result(timeout = rospy.Duration(10))
        rospy.loginfo("Sending the robot to its init pose, in case something is still in move_base queue")
        print "Resetting!!!!!!"
        
        for i in range(10):
            self.__perform_action([0, 0])
            self.rate.sleep()
        rospy.sleep(rospy.Duration(2))
        self.RL.reset()
        self.trial_begin = rospy.Time().now().to_time()
        self.running = True
    
    def expand_state(self, state):
        # Add additional information to the image
        bumpers_state = [self.fbumper, self.bbumper, self.lbumper, self.rbumber]
        bumpers_state = numpy.asarray(bumpers_state)
        
        if self.method == "cacla":
            # robot_state = numpy.hstack((bumpers_state, self.imu, \
            #                        numpy.asarray([self.linear_speed, self.angular_speed, self.previous_action[0], self.previous_action[1]])))
            # Without IMU
            robot_state = numpy.hstack((bumpers_state, \
                                numpy.asarray([self.linear_speed, self.angular_speed, self.previous_action[0], self.previous_action[1]])))
        else:
            # print self.previous_action
            robot_state = numpy.hstack((bumpers_state, \
                                numpy.asarray([self.linear_speed, self.angular_speed, self.previous_action])))
            
        # print robot_state.shape
        # print state.shape
        # self_state = numpy.hstack(())
        return numpy.hstack((state, robot_state))        
            
    def check_collision(self):
        if max(self.chassis, self.fbumper, self.bbumper, \
               self.lbumper, self.rbumber) == self.bumper_value:
            self.in_collision = True
        else:
            self.in_collision = False
                       
    def chassiscb(self, data):
        if data.data:
            self.chassis = self.bumper_value;
        else:
            self.chassis = -self.bumper_value;

    def save_and_exit(self):

        fig1 = np.asarray(self.fig1_data)
        fig2 = np.asarray(self.fig2_data)
        succ = np.asarray(self.consecutive_succes)
        actions = np.asarray(self.actions_numbers)
        try:
            fold = str(self.x_init) + '_' + str(self.y_init) + '/'
            os.mkdir(self.base_path + fold)
        except:
            print "folder {} already exist".format(self.base_path + fold)
        np.save(self.base_path + fold + 'fig1', fig1)    
        np.save(self.base_path + fold + 'fig2', fig2)
        np.save(self.base_path + fold + 'success', succ)
        np.save(self.base_path + fold + 'actions', actions)
        np.save(self.base_path + fold + 'start', np.array([self.x_init, self.y_init]))
        
    def save_state(self, force = False):
        if (time.time() - self.save_timer > self.save_interval) or force:
            base_path = os.path.join(self.base_path)
            filepath = os.path.join(base_path, 'stage_single_goal_caferoom_state')
            print "Saving maze_multi_goal_caferoom_state  States to File..."
            try:
            
                f = file(filepath, 'wb')
                dill.dump(self.__dict__, f)
                
                print "Saving Succeeded"
            except Exception as ex:
                print ex
            self.save_timer = time.time()
            
    def load_state(self):
        base_path = os.path.join(self.base_path)
        filepath = os.path.join(base_path, 'stage_single_goal_caferoom_state')
        
        try:
            f = file(filepath, 'rb')
            self.__dict__.update(dill.load(f))
        except Exception as ex:
            print "Loading of states went wrong"
            print ex
        
    def odomcb(self):
        
        global NUM1, NUM2, LOCK1, LOCK2, ARR1, ARR2
        # self.linear_speed = data.twist.twist.linear.x * self.speed_multiplier
        # self.angular_speed = data.twist.twist.linear.z * self.speed_multiplier
        self.linear_speed = 0
        self.angular_speed = 0
        
        # Switched from Callback to wait_for_message
        data = rospy.wait_for_message('/odom', Odometry)
        
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        
        rotation = (data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                   data.pose.pose.orientation.z, data.pose.pose.orientation.w) 
        
        degrees = tf.transformations.euler_from_quaternion(rotation)
        self.degrees = math.degrees(degrees[2])
        self.sin_theta = math.sin(degrees[2])
        self.cos_theta = math.cos(degrees[2])
        self.radian = degrees[2]
        
        try:
            self.loop_rate.sleep()
        except ROSTimeMovedBackwardsException as e:
            # Ros Time backward due to restarting stage. Ignoring.
            print repr(e)
            rospy.logwarn("Loop rate move backward detected. Ignoring")
        if self.running:
            self.steps_number += 1
            terminal = False
            skip_update = False
            
            state, pos = self.get_state_representation(None, output = self.input_type)
            action = self.RL.select_action(state)
            if self.method == "cacla":
                self.__perform_action(action)  # Gives absolute speed values
            else:
                if (not self._perform_movebase_action_no_rot(action, pos)):
                    self.failed_action_queue.append(1)
                else:
                    self.failed_action_queue = []
            rospy.sleep(0.2)    
            updated_pos = rospy.wait_for_message('/odom', Odometry)
            updated_pos = rospy.wait_for_message('/odom', Odometry)
            updated_pos = rospy.wait_for_message('/odom', Odometry)
            updated_pos = rospy.wait_for_message('/odom', Odometry)
            self.x = updated_pos.pose.pose.position.x
            self.y = updated_pos.pose.pose.position.y    
            next_state, pos = self.get_state_representation(None, output = self.input_type)
                    
            reward = self.get_reward()

            times_up = self.steps_number >= self.trial_max_steps
            if reward >= self.angular_fix_reward: 
                terminal = True
                self.success_trials += 1
                self.RL.update_online(state, action, reward, self.empty_vector)
            elif times_up:
                self.failed_trials += 1
                terminal = True
                self.RL.update_online(state, action, reward, next_state)
            else:
                self.RL.update_online(state, action, reward, next_state)
                 
            # self.state_history.append(state,reward,terminal,skip_update)
            self.previous_action = action
            
            rospy.logdebug(" Reward: %3.2f, action %s" % (reward , action))
           
            if self.RL.total_actions > self.replay_frequency_action:
                self.replay_frequency_action += self.replay_frequency_step
                rospy.loginfo("EXPERIENCE__REPLAY")
                self.RL.update_networks_plus(experience_replay = True)
                rospy.loginfo("EXPERIENCE__REPLAY_FINISHED")

                self.fig1_data.append([self.RL.total_actions,
                                self.RL.cum_reward / self.train_frequency_step])
                
            if self.RL.total_actions % self.train_frequency_action == 0:
                self.reward_data.append((self.RL.total_actions, self.RL.cum_reward / self.train_frequency_step))
                self.RL.cum_reward = 0
                    
            if self.RL.total_actions % self.figure_update_frequency == 0:
                ARR1 = self.reward_data
                ARR1 = [(0, 0), (1, 1), (2, 2), (3, 3)]
                NUM1.value = 0
                
                ARR2 = self.steps_data
                ARR2 = [(0, 0), (1, 1), (2, 2), (3, 3)]
                NUM2.value = 0
                
            if len(self.failed_action_queue) > 10:
                print "10 Actions Failed. ROBOT IS STUCK. Restarting Trial."
                self.failed_trials += 1
                times_up = True
                self.failed_action_queue = []
                            
            if times_up or reward >= self.angular_fix_reward:
            
            # self.success_trials += 1
            # if True:

                self.actions_numbers.append(self.steps_number)
                if len(self.actions_numbers) > 200:
                    self.actions_numbers = self.actions_numbers[1:]
                
                self.fig2_data.append([self.RL.progress, self.RL.steps])

                success = float(self.success_trials) / float(self.success_trials + self.failed_trials)
                # success = 1
                self.consecutive_succes.append(success)
                if len(self.consecutive_succes) > 15:
                    self.consecutive_succes = self.consecutive_succes[1:]
                
                if (self.failed_trials == 0 and self.success_trials == 0):
                    pass  # This happens after a batch update
                else:
                    print "SUCCESS RATIO: ", float(self.success_trials) / float(self.success_trials + self.failed_trials)
                    # pass
                self.running = False
                self.RL.save()
                self.save_state()
                
                self.steps_data.append((self.RL.progress, self.RL.steps))
                
                try:
                    self.total_reward_queue.put_nowait(self.RL.total_reward)
                except Queue.Full:
                    # self.analyze_experiment()
                    pass

                if np.asarray(self.consecutive_succes).mean() >= 0.80 and self.steps_number < 150:
                    print 'learned!!'
                    print 'now exiting'
                    self.save_and_exit()
                    self._prepare_next_goal()
                    # exit()
                    rospy.sleep(10)  

                print 'consecutive successes: ', self.consecutive_succes
                print ' steps numbers:        ', self.steps_number
                self.reset()
                self.steps_number = 0
    
    def get_state_representation(self, image, output = "location"):
        ''' Uses the neural network to extract state representation'''
        if output == "autoencoder":  # Only use the encoder part
            return self.encode(image)[0]
        elif output == "location":  # Use the location estimations
            out = self.encode_n_estimate(image)[0]
            final = scale(out)
            # print "X:%s, Y:%s" % (final[0], final[1])
            return out
        else:
            
            # res = normalize([self.x, self.y, self.sin_theta, self.cos_theta])
            res = self._convert_to_grid(self.x, self.y, self.radian)
            
            return res
        
        return state
    
    def get_reward(self):
        reward = 0

        def calculate_goal_distance():
            now = rospy.Time().now()
            try:
                self.transform_listener.waitForTransform(self.reference_frame, "base_link", now, rospy.Duration(0.1))
                translation, rotation = self.transform_listener.lookupTransform(self.reference_frame, "base_link", now) 
            except tf.Exception, e:
                print repr(e)
                return
            
            orientation = tf.transformations.euler_from_quaternion(rotation)
            theta = orientation[2] / (math.pi / 180.0)
            # x = translation[0]
            # y = translation[1]
            # print 'Goal is : ', self.goal
            # print 'Pos is  : ', (self.x, self.y)
            
            d_angle = math.fabs(theta - self.goal['theta'])
            d_location = math.fabs(self.goal['x'] - self.x) + abs(self.goal['y'] - self.y)
            rospy.loginfo("Location Diff: %f", d_location)
            if d_location <= 0.5:
                rospy.loginfo("Angle Diff: %f", d_angle)
            return d_location, d_angle
        
        if self.in_collision:
            rospy.logdebug("in collision")
            self.in_collision = False
            return self.obstacle_punishment
        
        if self.linear_speed <= 0:
            reward += self.negative_speed_punishment
        
        try:
            d_location, d_angle = calculate_goal_distance()
        except:
            return reward
        
        # Give reward based on distance
        '''
        if d_location <= 1.0 and d_location > self.goal_xy_threshold:
            reward += 2
        elif d_location <= 1.5 and d_location > 1.0:
            reward += 1
        elif d_location <= 2.0 and d_location > 1.5 :
            reward += 0.5
        elif d_location <= 3.0 and d_location > 2.0:
            reward += -0.2
        elif d_location > 3.0:
            reward += -1.0
        '''
        # Check location distance for reward
        if d_location <= self.goal_xy_threshold:
            rospy.loginfo("Goal Reached")
            reward = self.angular_fix_reward
            return reward
        elif d_location <= self.goal_xy_linear_threshold and self.complex_reward:
            reward += self.ll_reward(d_location)
        else:
            return reward + self.fix_punishment
        
        # If robot was in good location, check for angular precision
        if d_location <= self.goal_xy_threshold:
            if d_angle <= self.goal_theta_threshold:
                # TODO: Added Stopping Criteria for final reward. Has Magic Numbers. Fix it
                if self.stop_at_goal: 
                    if self.linear_speed < 0.05 and self.angular_speed < 0.05:
                        reward = self.angular_fix_reward
                    else:
                        reward += 2
                else:
                    rospy.loginfo("Goal Reached")
                    reward = self.angular_fix_reward
            elif d_angle <= self.goal_theta_linear_threshold and self.complex_reward:
                reward += self.ld_reward(d_angle)
            else:
                return reward
            
        return reward
    
    def spin(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print "Shutting down"

'''
This File is responsible to run the single goal 2D simulator for big apartment RL experiments.
You HAVE to use a trained network otherwise this will TAKE FOR EVER.

The results will be made in the same path here. If you want to parallelize the process on multiple PCs, you can
select the indices below. I do not recommend it since it defeats the multi goal purpose.
'''
if __name__ == '__main__':
    '''
    if os.environ['HOME'] == "/home/amir":
        
        base_path = "/home/amir/"
    elif os.environ['HOME'] == "/home/borg":
        
        base_path = "/home/borg/nfq/"
    else:
        base_path = "/media/amir/Datastation/nav-data-late14/BACKUPSTUFF"
    
    
    
      
    p1 = Process(target=figure_make.update, args=(LOCK1, NUM1, ARR1))
    p2 = Process(target=figure_make.update2, args=(LOCK2, NUM2, ARR2))
    p1.start()
    p2.start()
    '''
    
    reference_frame = 'odom'

    simulation_env = "g_start.sh"
    print "RUNNING THE %s ENVIRONMENT" % simulation_env
 
    path = '../goals_starts/'
    goals = np.load(path + 'goals.npy')
    starts = np.load(path + 'starts.npy')
    maze = np.load(path + 'approx.npy')
    indeces = np.load(path + 'indeces.npy')
    # for ind, (x, y) in enumerate(goals):

    print 'starting positions'
    print starts
    time.sleep(1)
    ###################################################
    # Here specify which goal you want to use
    inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ###################################################
    goal_list = []
    x_init_list = []
    y_init_list = []
    goal_list_idx = []
    last_index_list = []
    
    ordered_index_list = []
    for ind in inds:
    
        x, y = goals[ind]
        
        print 'goal is : ', (x, y)

        fold = path + 'goal_' + str(ind) + '/'
        tmp = copy.deepcopy(maze)
        try:
            os.mkdir(fold)
        except:
            pass
        try:
            os.mkdir(fold + 'nfq')
        except:
            pass
        ind1, ind2 = indeces[ind]
        tmp[ind1, ind2] = 2
        
        f3 = plt.figure(3)
        plt.matshow(tmp, fignum = 3)
        f3.savefig(fold + 'maze.png')
        plt.close('all')
        
        for ind2, (x_init, y_init) in enumerate(starts):
            # x_init, y_init = (-8.1,-2.1)
            goal_list.append((x, y, 0))
            x_init_list.append(x_init)
            y_init_list.append(y_init)
            # last_index_list.append(ind2)
            ordered_index_list.append((ind, x_init, y_init))

    experiment = CACLA_nav(path, reference_frame, x_init_list, y_init_list, goal_list, ordered_index_list)
    experiment.load_state()
    while not rospy.is_shutdown():
        try:
            if experiment.running:
                experiment.odomcb()
            else:
                pass
        except rospy.ROSInterruptException as e:
            print repr(e)
            break
    os.system('killall xterm -9')    
    
