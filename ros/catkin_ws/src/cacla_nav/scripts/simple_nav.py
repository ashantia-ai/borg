#!/usr/bin/env python
import matplotlib.pyplot as plt
import rospy, roslib

import cPickle, sys
import math, numpy, cv2

import rl_methods.nfq as nfq

import theano
from theano import tensor as T, config, shared
from theano.tensor.shared_randomstreams import RandomStreams



import sensor_msgs
import rl_methods.nfq 

from sensor_msgs.msg import Image, CameraInfo, Imu
from cacla_nav.msg import visualize
from std_msgs.msg import Bool, Int8
from geometry_msgs.msg import Twist
from geometry_msgs.msg._Quaternion import Quaternion
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ContactsState
from robot_control.srv import *
from robot_control.msg import *
from move_base_msgs.msg import *


import actionlib

import tf

from cv_bridge import CvBridge, CvBridgeError

import multiprocessing
from multiprocessing import Lock, Process, Value, Array, Manager

import figure_make

import os
import gazebo
import Queue

manager = Manager()
NUM1 = Value('i', 1)
ARR1 = manager.list()
LOCK1 = Lock()

NUM2 = Value('i', 1)
ARR2 = manager.list()
LOCK2 = Lock()

def normalize(scale):    
    maxX = 7.3
    minY = -1.9
    maxY = 2.57 + -minY
    
    # normalize the x 
    scale[0] /= maxX
    
    #normalize the y
    
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
    def __init__(self, base_path, reference_frame, **kwargs):
        
        #Prepares Action-Service for NFQ
        
        self.input_type = "ground_truth" #Options: 1- autoencoder . 2-location, 3- ground truth location, Default: ground truth
        self.method = "nfq" #1- Cacla 2- NFQ
        
        self.steps_number = 1
        self.trial_max_steps = 1000
        
        self.__init_ros_node()
        self.__prepare_action_service()
        
        self.__prepare_file_structure(base_path)
        
        self.reference_frame = reference_frame
        self.transform_listener = tf.TransformListener()
        self.bridge = CvBridge()
        
           
        self.__prepare_init_n_goal_position()
        self.__prepare_rewards()
        
        self.__adjust_sensor_multipliers()
        self.__prepare_action_variables()
        
        self.__prepare_performance_variables()
        self.__prepare_experience_replay()
        
        #self.imu_test = numpy.zeros((1,5))
        #self.imu_count = 0
        
        self.steps_data = []
        self.reward_data = []
        self.figure_update_frequency = 1
        
        
        
        self.nfq_args = {'epsilon':(0.1,0.1) \
                ,'discount_factor':0.9, 'random_decay': 0.995\
                , 'explore_strategy':3, 'activation_function':1}
        
        
        if kwargs:
            self.set_parameters(**kwargs)
        
        self.__convert_to_grid(self.x, self.y, self.degrees) #Initializes empty vector size
        self.__prepare_RL_variables()
        
        
        self.reset()
        
        self.__init_subscribers()
        
    
    def __repr__(self, *args, **kwargs):
        return object.__repr__(self, *args, **kwargs)
    
    def __adjust_sensor_multipliers(self):
        if self.input_type == "autoencoder": 
            self.bumper_value = 5 #This value will be sent to NN, it is higher than one to show importance
            self.imu_multiplier = 5 #This value will be multiplied by imu values, it is higher than one to show importance
            self.speed_multiplier = 5 #This value will be multiplied by speed values, it is higher than one to show importance
        else:
            self.bumper_value = self.imu_multiplier = self.speed_multiplier = 1 #There are only location values, so no multiplying
        
        self.fbumper = self.bbumper = self.lbumper = self.rbumber = self.chassis = -self.bumper_value
        self.in_collision = False
        self.linear_speed = self.angular_speed = 0
        self.imu = numpy.zeros((1,5))
    def __convert_from_grid(self, x, y, theta):
        minX = 0
        minY = -1.9
        
        
        y += -minY
        metric_resolution = 0.5
        angular_resolution = 90
        
        precise_x = x * metric_resolution
        precise_y = y * metric_resolution
        precise_theta = theta * angular_resolution 
    def __convert_to_grid(self, x, y, theta, x_len = 8, y_len = 5):
        '''
        Converts the location output to grid like format
        '''
        
        minX = 0
        minY = -1.9
        
        
        y += -minY
        metric_resolution = 0.5
        angular_resolution = 90
        
        columns = math.ceil(x_len / metric_resolution)
        rows = math.ceil(y_len / metric_resolution)
        depth = math.ceil( 360 / angular_resolution)
        

        
        x_idx = int(x / metric_resolution)
        y_idx = int(y / metric_resolution)
        d_idx = int(theta / angular_resolution)
        
        #print "x: %s, y: %s, theta %s" % (x_idx,y_idx, d_idx)
        
        pos_matrix = numpy.zeros((columns, rows, depth), dtype=theano.config.floatX)
        d_matrix = numpy.zeros((columns, rows), dtype=theano.config.floatX)
        pos_matrix[x_idx, y_idx, d_idx] = 1
        d_matrix[x_idx, y_idx] = 1
        
        dest = cv2.resize(d_matrix, dsize = (0,0),fx = 10, fy = 10)
        
        cv2.imshow('location', dest)
        cv2.waitKey(1)
        
        self.empty_vector = numpy.zeros((1, pos_matrix.size))
        self.empty_vector = numpy.reshape(self.empty_vector, -1)
        
        return numpy.reshape(pos_matrix, -1)
              
    def __create_directories(self):
        if not os.path.exists(self.base_path):
            raise Exception("Base path %s doesn't exist" % (self.base_path))
        elif os.path.exists(self.method_path):
            try:
                os.mkdir(self.method_path)
            except Exception, e:
                print repr(e)
                
            
    
    
    
    def __init_ros_node(self):
        ''' Initialized the ros node and the main publishers
        '''
        rospy.loginfo("Callbacks registered")
        rospy.init_node('Simple_nav')
        rospy.loginfo("Node initialized")
        
        self.cacla_pub = rospy.Publisher('cacla_actions', visualize, queue_size = 1)
        self.action_pub = rospy.Publisher('rl_actions', Int8, queue_size = 1)
        self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size = 1)
        
        self.client = actionlib.SimpleActionClient('readaction', robot_control.msg.rlAction)
        self.movebase_client = actionlib.SimpleActionClient('readaction', MoveBaseAction)
        self.client.wait_for_server(3)
        self.movebase_client.wait_for_server()
        
        #Frequency of the Imagecb function
        self.loop_rate = rospy.Rate(100)
        #Used to make sure robot will stop before reseting the trial
        self.rate = rospy.Rate(100)
        
    def __init_subscribers(self):
        rospy.Subscriber("/contact_sensor_plugin/collision", Bool, self.chassiscb, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self.odomcb, queue_size=1)
          
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
        
        #print "Linear Speed: ", linear_speed, " Angular Speed: ", angular_speed
        # create a twist message, fill in the details
        twist = Twist()
        twist.linear.x = linear_speed# our forward speed
        twist.linear.y = 0; twist.linear.z = 0;     # we can't use these!        
        twist.angular.x = 0; twist.angular.y = 0;   #          or these!
        twist.angular.z = angular_speed;                        # no rotation
        
        self.publisher.publish(twist)
    
    def __perform_jump_action(self, action_num):
        epsilon = 0.2
        z = roll = pitch = 0
        x = self.x
        y = self.y
        theta = self.degrees
        step_size = 0.5
        
        if action_num == 0:
            if math.fabs(theta - 0) < epsilon:
                x -= step_size
            elif math.fabs(theta - 1.57) < epsilon:
                y -= step_size
            elif math.fabs(theta - 3.14) < epsilon:
                x += step_size
            elif math.fabs(theta - 4.71) < epsilon:
                y += step_size
            else:
                raise
        
        elif action_num == 1:
            if math.fabs(theta - 0) < epsilon:
                x += step_size
            elif math.fabs(theta - 1.57) < epsilon:
                y += step_size
            elif math.fabs(theta - 3.14) < epsilon:
                x -= step_size
            elif math.fabs(theta - 4.71) < epsilon:
                y -= step_size
            else:
                raise
        elif action_num == 2:
            theta += 1.57
            if theta >= 6.28:
                theta -= 6.28
            
        elif action_num == 3:
            theta -= 1.57
            if theta < -0.1:
                theta += 6.28
                
        gazebo.move_to(self.gazebo_model, x, \
                       y, \
                       z, \
                       roll, \
                       pitch, \
                       theta)
        
        
    def __perform_action(self, action):
        def scale_action(action):
            pass
        #TODO: add max , min speed limitation. BUT NOT HERE, in CACLA 
        linear_speed, angular_speed = action
        
        #print "Linear Speed: ", linear_speed, " Angular Speed: ", angular_speed
        # create a twist message, fill in the details
        twist = Twist()
        twist.linear.x = linear_speed# our forward speed
        twist.linear.y = 0; twist.linear.z = 0;     # we can't use these!        
        twist.angular.x = 0; twist.angular.y = 0;   #          or these!
        twist.angular.z = angular_speed;                        # no rotation
        
        self.publisher.publish(twist)
    
    def __perform_discreet_action(self, action_num):
        #return 
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
        #print "result is:", response.result
        
    def __perform_movebase_action(self, action_num):
        
        test = rospy.Time.now()
        goal = MoveBaseGoal()
        
        if action_num == 0:
            x_change = self.x 
            self.__convert_to_grid
        elif action_num == 1:
            pass
        elif action_num == 2:
            pass
        elif action_num == 3:
            pass
        elif action_num == 4:
            pass
        
        goal.target_pose.pose.position.x = self.x
        goal.target_pose.pose.orientation.w = 1.0
        goal.target_pose.header.frame_id = 'first_move'
        goal.target_pose.header.stamp = rospy.Time.now()
        
        self.movebase_client.send_goal(goal)
        
        finished_before_timeout = self.client.wait_for_result(rospy.Duration.from_sec(10))
        if finished_before_timeout:
            rospy.logdebug("Action Time: %s" % (rospy.Time.now().to_sec() - test.to_sec()))
        else:
            rospy.logwarn("Action didn't finish")
            self.movebase_client.cancel_all_goals()
        
        return self.movebase_client.get_result()
        #print "result is:", response.result
        

        #TODO:change to service
    def __prepare_action_service(self):
        pass
    def __prepare_action_variables(self):
        #TODO: needs to called from set_params too
        #Action Variables
        self.maximum_linear_speed = 0.4 #m/s
        self.maximum_angular_speed = 1.57 # rad/s
        self.linear_acc_step = 0.04
        self.angular_acc_step = 1
        self.last_linear_speed = 0
        self.last_angular_speed = 0
        if self.method == "cacla":
            self.previous_action = [0,0]
        else:
            self.previous_action = 0
    def __prepare_experience_replay(self):
        self.state_history = []
        self.success_history = []
        self.replay_frequency = 100
        self.train_frequency = 10
        self.train_frequency_action = 100
        self.train_frequency_step = 100
        self.replay_frequency_action = 1000
        self.replay_frequency_step = 1000
               
    def __prepare_file_structure(self,base_path):
        self.base_path = base_path
        self.method_path = os.path.join(self.base_path, self.method)
        
        self.__create_directories()
        
    def __prepare_init_n_goal_position(self):
        #Name of the model to control through code
        self.gazebo_model = "sudo-spawn"
        
        self.sudo_initial_position = Position(2.5, -0.5, 0.02, 0,0,math.radians(0)) #next to curtain side counter, facing down
        #self.sudo_initial_position = Position(2.9, 0.04, 0.02, 0, 0, -1.57) #next to curtain side counter, facing left
        #self.sudo_initial_position = Position(2.53, 0.52, 0.02, 0, 0, -1.57) #next to curtain side counter, by the choke hole, ficing left
        #self.sudo_initial_position = Position(4.8, -0.852, 0.02, 0, 0, 3.14) #next to fridge, facing up
        #self.sudo_initial_position = Position(4, -0.5, 0.02, 0,0,math.radians(0)) #Right on goal
        
        self.x = self.sudo_initial_position.x
        self.y = self.sudo_initial_position.y
        self.sin_theta = math.sin(self.sudo_initial_position.yaw)
        self.cos_theta = math.cos(self.sudo_initial_position.yaw)
        self.degrees = math.degrees(self.sudo_initial_position.yaw)
        
        #Initial Position update after a successfull learning
        self.yaw_change = 90 #in degrees
        self.x_change = 50 #in cm
        self.y_change = 50 #in cmF
        
        #Goal Parameters
        self.goal_list = []
        self.goal_theta_threshold = 50 #Degrees
        self.goal_theta_linear_threshold = 20 #Degrees
        self.goal_xy_threshold = 0.5 #Meters
        self.goal_xy_linear_threshold = 0.3 #Meters
        
        #self.goal = {'x':3.66, 'y':-0.59, 'theta':-90} #Next to stove, facing to it
        self.goal = {'x':4, 'y':-0.5, 'theta':0.02} # infront of init position.
    
    def __prepare_performance_variables(self):
        
        self.queue_size = 10
        self.total_reward_queue = Queue.Queue(self.queue_size)
        
        self.acceptable_run_threshold = 50 #Average total reward of 100 runs should be bigger than 50
        self.success_trials_threshold = 10
        self.success_ratio_threshold = 0.8 #The winning ratio required for updating the initial position
        self.success_trials = 0.0 
        self.failed_trials = 0.0
        self.last_position_update = 0
        self.fail_trial_threshold = 3000 #Number of failed trials before sigma is resetted to a higher value
        self.min_sigma_update = 0.3 #Only reset sigma if it is smaller than this value
        self.win_ratio = 0
        
    def __prepare_rewards(self):
        #The reward if location of the robot is close to goal
        self.location_fix_reward = 0.5
        #The reward if also the angle is correct
        self.angular_fix_reward = 100
        
        #Negative reward per time step
        self.fix_punishment = -0.1
        #Punishment for not reaching the goal
        self.trial_fail_punishment = -0.1
        #Punishmend for hitting an obstacle
        self.obstacle_punishment = -2.0
        self.negative_speed_punishment = -0.0
        
        self.ll_reward = lambda i: -8 * i + 9 #linear location formula - 1 meter is limit
        self.ld_reward = lambda i: -0.2 * i + 7 #linear angular formula - 30 degrees is limit
        self.complex_reward = False #Enables/Disables smooth reward reduction aroud the goal
        self.stop_at_goal = False
        
    def __prepare_RL_variables(self):
        action_outputs = 2
        nfq_action_outputs = 4
        
        if self.method == ("cacla"): 
            
            self.RL = util.cacla.CACLA(self.method_path, action_outputs, **self.cacla_args)
        elif self.method == ("nfq"):
            self.RL = rl_methods.nfq.NFQ(self.method_path, nfq_action_outputs, self.empty_vector.size, **self.nfq_args)
        else:
            self.RL = rl_methods.nfq.NFQ(self.method_path, nfq_action_outputs, **self.nfq_args)
        
        self.trial_length = 1000 #seconds
        self.trial_begin = rospy.Time().now().to_time()
        self.trial_time = lambda : rospy.Time().now().to_time() - self.trial_begin 
        self.running = True
        
        
        
    def __visualize__(self, action, explore):
        actions = visualize()
        
        actions.action.linear.x = action[0]
        actions.action.angular.z = action[1]
        
        actions.explore_action.linear.x = explore[0]
        actions.explore_action.angular.z = explore[1]
        
        self.cacla_pub.publish(actions)
        
        
                
    def analyze_experiment(self):
        rewards = []
        f = open(os.path.join(self.method_path, "analysis.txt"),'a')
        while True:
            try:
                rewards.append(self.total_reward_queue.get_nowait())
            except Queue.Empty:
                average_run = numpy.mean(numpy.asarray(rewards))
                report = self.get_current_status(average_run)
                
                
                self.win_ratio = float(self.success_trials)/float(self.success_trials + self.failed_trials)
                print "SUCESS RUNS FOR THIS ANALYSIS IS %f" % (self.win_ratio)
                report += "Current Win Ration: %f \n" % (self.win_ratio)
                if  (False and average_run >= self.acceptable_run_threshold) or self.win_ratio >= self.success_ratio_threshold:
                    self.last_position_update = self.RL.progress
                    self.success_trials = 0
                    self.failed_trials = 0
                    report += "Learning done. Updating the initial Position\n"
                    report += "--------------------------------------------\n"
                    self.update_initial_position(0,0,self.yaw_change)
                if self.RL.progress - self.last_position_update > self.fail_trial_threshold:
                    if self.RL.sigma < self.min_sigma_update:
                        self.RL.sigma = self.cacla_args['sigma'][1] #Resetting Sigma to high value again
                        report += "*******SIGMA IS RESETED*********************\n"
                     
                break
        f.write(report)
        f.close()
    def get_current_status(self, score = 0):
        status =  "------ Iteration No %s ------\n" % (self.RL.progress)
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
        self.client.cancel_goals_at_and_before_time(rospy.Time.now())
        self.client.cancel_all_goals()
        for i in range(100):
            self.__perform_action([0,0])
            self.rate.sleep()
        gazebo.move_to(self.gazebo_model, self.sudo_initial_position.x, \
                       self.sudo_initial_position.y, \
                       self.sudo_initial_position.z, \
                       self.sudo_initial_position.roll, \
                       self.sudo_initial_position.pitch, \
                       self.sudo_initial_position.yaw)

        self.RL.reset()
        self.trial_begin = rospy.Time().now().to_time()
        self.running = True
        
    
    def expand_state(self, state):
        #Add additional information to the image
        bumpers_state = [self.fbumper, self.bbumper, self.lbumper, self.rbumber]
        bumpers_state = numpy.asarray(bumpers_state)
        
        if self.method == "cacla":
            #robot_state = numpy.hstack((bumpers_state, self.imu, \
            #                        numpy.asarray([self.linear_speed, self.angular_speed, self.previous_action[0], self.previous_action[1]])))
            #Without IMU
            robot_state = numpy.hstack((bumpers_state, \
                                numpy.asarray([self.linear_speed, self.angular_speed, self.previous_action[0], self.previous_action[1]])))
        else:
            #print self.previous_action
            robot_state = numpy.hstack((bumpers_state, \
                                numpy.asarray([self.linear_speed, self.angular_speed, self.previous_action])))
            
        #print robot_state.shape
        #print state.shape
        #self_state = numpy.hstack(())
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
               
        
        
        
    def odomcb(self, data):
        
        global NUM1, NUM2, LOCK1, LOCK2, ARR1, ARR2
        self.linear_speed = data.twist.twist.linear.x * self.speed_multiplier
        self.angular_speed = data.twist.twist.linear.z * self.speed_multiplier
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        
        rotation = (data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                   data.pose.pose.orientation.z, data.pose.pose.orientation.w) 
        
        degrees = tf.transformations.euler_from_quaternion(rotation)
        self.degrees= math.degrees(degrees[2])
        self.sin_theta = math.sin(degrees[2])
        self.cos_theta = math.cos(degrees[2])
        
        #self.loop_rate.sleep()
        if self.running:
            self.steps_number += 1
            terminal = False
            skip_update = False
            
            state = self.get_state_representation(None, output = self.input_type)
            action = self.RL.select_action(state)
            #action = 0 
            if self.method == "cacla":
                self.__perform_action(action) #Gives absolute speed values
            else:
                self.__perform_jump_action(action)
                #self.__perform_discreet_action(action)
                pass
                
            next_state = self.get_state_representation(None, output = self.input_type)
            self.check_collision()         
            reward = self.get_reward()
            
            flipped = math.fabs(degrees[0] - 1.0 > 0) or math.fabs(degrees[1] - 1.0 > 0) 
            if flipped:
                print "I am flipped. Restarting"
            times_up = self.steps_number >= self.trial_max_steps or flipped 
            if reward >= self.angular_fix_reward: 
                print "GOAL REACHED"
                terminal = True
                self.success_trials += 1
                self.RL.update_online(state, action, reward, self.empty_vector)
            elif times_up:
                print "TIMES UP"
                self.failed_trials += 1
                terminal = True
                self.RL.update_online(state, action, reward, next_state)
            else:
                self.RL.update_online(state, action, reward, next_state)
                
                 
            #self.state_history.append(state,reward,terminal,skip_update)
            self.previous_action = action
            
            rospy.logdebug(" Reward: %3.2f, action %s" %( reward ,action))
            
            if self.RL.total_actions % self.replay_frequency_action == 0:
                self.RL.update_networks_plus(experience_replay = True)
                
            if self.RL.total_actions % self.train_frequency_action == 0:
                self.reward_data.append((self.RL.total_actions, self.RL.cum_reward / self.train_frequency_step))
                self.RL.cum_reward= 0
                    
            if self.RL.total_actions % self.figure_update_frequency == 0:
                ARR1 = self.reward_data
                ARR1 = [(0,0), (1,1), (2,2),(3,3)]
                NUM1.value = 0
                
                ARR2 = self.steps_data
                ARR2 = [(0,0), (1,1), (2,2),(3,3)]
                NUM2.value = 0
                            
            if times_up or reward >= self.angular_fix_reward:
                
                self.steps_number = 0
                
                if (self.failed_trials == 0 and self.success_trials == 0):
                    pass #This happens after a batch update
                else:
                    #print "SUCCESS RATIO: ", float(self.success_trials)/float(self.success_trials + self.failed_trials)
                    pass
                self.running = False
                self.RL.save()
                
                self.steps_data.append((self.RL.progress, self.RL.steps))
                
                try:
                    #self.total_reward_queue.put_nowait(self.RL.total_reward)
                    pass
                except Queue.Full:
                    #self.analyze_experiment()
                    pass
                   

                self.reset()
        
    
    def get_state_representation(self, image, output = "location"):
        ''' Uses the neural network to extract state representation'''
        if output == "autoencoder": #Only use the encoder part
            return self.encode(image)[0]
        elif output == "location": #Use the location estimations
            out = self.encode_n_estimate(image)[0]
            final = scale(out)
            #print "X:%s, Y:%s" % (final[0], final[1])
            return out
        else:
            
            #res = normalize([self.x, self.y, self.sin_theta, self.cos_theta])
            res = self.__convert_to_grid(self.x, self.y, self.degrees)
            
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
            x = translation[0]
            y = translation[1]
            d_angle = math.fabs(theta - self.goal['theta'])
            d_location = (abs(self.goal['x'] - x) + abs(self.goal['y'] - y))
            
            return d_location, d_angle
        
        if self.in_collision:
            return self.obstacle_punishment
        
        if self.linear_speed <= 0:
            reward += self.negative_speed_punishment
        
        try:
            d_location, d_angle = calculate_goal_distance()
        except:
            return reward
        #Check location distance for reward
        if d_location <= self.goal_xy_threshold:
            reward += self.location_fix_reward
        elif d_location <= self.goal_xy_linear_threshold and self.complex_reward:
            reward += self.ll_reward(d_location)
        else:
            return reward + self.fix_punishment
        
        #If robot was in good location, check for angular precision
        if d_location <= self.goal_xy_threshold:
            if d_angle <= self.goal_theta_threshold:
                #TODO: Added Stopping Criteria for final reward. Has Magic Numbers. Fix it
                if self.stop_at_goal: 
                    if self.linear_speed < 0.05 and self.angular_speed < 0.05:
                        reward = self.angular_fix_reward
                    else:
                        reward += 2
                else:
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
        
if __name__ == '__main__':
    
    if os.environ['HOME'] == "/home/work-pc":
        
        base_path = "/home/work-pc/nav-data"
    elif os.environ['HOME'] == "/home/borg":
        
        base_path = "/home/borg/nav-data"
    else:
        base_path = "/media/amir/Datastation/nav-data-late14/BACKUPSTUFF"
    reference_frame = 'odom'
    
    
    '''    
    p1 = Process(target=figure_make.update, args=(LOCK1, NUM1, ARR1))
    p2 = Process(target=figure_make.update2, args=(LOCK2, NUM2, ARR2))
    p1.start()
    p2.start()
    '''
     
    experiment = CACLA_nav(base_path, reference_frame)
    experiment.spin()
    experiment.reset()
    

    
