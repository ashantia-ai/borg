#!/usr/bin/env python
import matplotlib.pyplot as plt
import rospy, roslib
from rospy import ROSTimeMovedBackwardsException

import cPickle, sys
import math, numpy, cv2
import numpy as np
import copy

import rl_methods.nfq as nfq

import time
import theano
from theano import tensor as T, config, shared
from theano.tensor.shared_randomstreams import RandomStreams


import std_srvs.srv
import stage_ros.srv

import sensor_msgs
import rl_methods.nfq 
import rl_methods.mgnfq as mgnfq

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


from multi_goal import Position

import os
import Queue
from numpy.f2py.auxfuncs import throw_error

from stage_nav import CACLA_nav

manager = Manager()
NUM1 = Value('i', 1)
ARR1 = manager.list()
LOCK1 = Lock()

NUM2 = Value('i', 1)
ARR2 = manager.list()
LOCK2 = Lock()


def scale(scale):    
    maxX = 7.3
    minY = -1.9
    maxY = 2.57 + -minY
    
    scale[0] *= maxX
    scale[1] *= maxY
    scale[1] -= -minY
    return scale


class MultiGoalStage(CACLA_nav):
    def __init__(self, base_path, reference_frame, x_init,y_init, goals, ordered_list_index = [(0,0,0)], g_l_for_parent = [], full_info = [],  **kwargs):

        
        self.running = False #Block callbacks to run
        self.mg_running = False
        super(MultiGoalStage, self).__init__( base_path, reference_frame, x_init,y_init, g_l_for_parent, ordered_list_index, **kwargs)
        self.nfq_args = {'epsilon':(0.1,0.1) \
                ,'discount_factor':0.9, 'random_decay': 0.995\
                , 'explore_strategy':3, 'activation_function':1
                , 'learning_rate':0.3}
        self._prepare_RL_variables()
        
        self.goals = goals
        self.goal_set = []
        self.full_info = full_info
        for x_g,y_g, x_ind, y_ind in self.goals:
            self.goal_set.append(Position(x=x_g, y=y_g, yaw = 0, x_ind = x_ind, y_ind = y_ind))
        self.reached_goals = [0] * len(self.goal_set)
        
        
        
        #Select below if goal and current state are part of input
        #ev_size = self.structure.size * 2
        #Select below if only current state is the input
        ev_size = 484
        self.empty_vector = numpy.zeros((1, ev_size), dtype = theano.config.floatX) 
        self.empty_vector = numpy.reshape(self.empty_vector, -1)
        
        #Which goal to follow! Default is the goal with highest td
        self.active_goal = 2
        self.base_path = os.path.join(self.lowest_path,'goal_'+str(self.active_goal)+'/')
        #which init position to start from
        self.init_index = 0
        self.init_no = len(self.full_info[0][1])
        self.RL.select_action(self.empty_vector, self.active_goal) #RL needs to be initialized first
        #self.active_goal = self.select_active_goal()    

        rospy.loginfo("First Active Goal is: %d", self.active_goal)

        self.steps_number = 1
        self.trial_max_steps = 200
        
        #Used for plotting
        self.n_steps_list = []
        self.actions_numbers = []
        self.cum_reward_list = []
        self.consecutive_succes=list(np.zeros(15))
        self.actions_numbers=list(np.zeros(200))
        self.fig1_data=[]       
        self.fig2_data=[]
        
        
        if kwargs:
            self.set_parameters(**kwargs)
            
        self.running = True #Allow callbacks to run
        self.mg_running = True
    def __reset_performance_variables(self):
        self.total_reward_queue = Queue.Queue(self.queue_size)
        self.success_ratio_threshold = 0.8 #The winning ratio required for updating the initial position
        self.success_trials = 0.0 
        self.failed_trials = 0.0
        self.last_position_update = 0
        self.win_ratio = 0  

        
    def _init_ros_node(self):
        ''' Initialized the ros node and the main publishers
        '''
        rospy.loginfo("Callbacks registered")
        rospy.init_node('Simple_nav')
        rospy.loginfo("Node initialized")
        
        self.cacla_pub = rospy.Publisher('cacla_actions', visualize, queue_size = 1)
        self.action_pub = rospy.Publisher('rl_actions', Int8, queue_size = 1)
        self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size = 1)
        
        self.movebase_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        #self.client.wait_for_server()
        self.movebase_client.wait_for_server()
        rospy.loginfo("Move_base client connected.")
        rospy.wait_for_service('reset_positions_with_pose', timeout = 10)
        self.reset_stage = rospy.ServiceProxy('reset_positions_with_pose', stage_ros.srv.resetToPose)
        pose = stage_ros.srv.resetToPoseRequest()
        pose.x = self.x_init
        pose.y = self.y_init
        self.reset_stage(pose)
        rospy.loginfo("Stage ROS is ready...")
        
        #Frequency of the Imagecb function
        self.loop_rate = rospy.Rate(100)
        #Used to make sure robot will stop before reseting the trial
        self.rate = rospy.Rate(100)
        
    def _init_subscribers(self):
        rospy.Subscriber("/collision", Bool, self.chassiscb, queue_size=1)
        #rospy.Subscriber("/base_pose_ground_truth", Odometry, self.odomcb, queue_size=1)
        
    def _initiate_batch_update(self):
        self.cum_reward_list.append((self.RL.total_actions, self.RL.cum_reward / self.train_frequency_step))
        #draw(1, self.cum_reward_list)
        self.RL.cum_reward= 0

        self.train_frequency_action += self.train_frequency_step 
        self.RL.update_networks_plus()
        
    def _prepare_file_structure(self,base_path):
        
        self.method_path = os.path.join(self.lowest_path, self.method)
        print "method path", self.method_path
        self._create_directories()
        
    def _create_directories(self):
        if not os.path.exists(self.lowest_path):
            raise Exception("Base path %s doesn't exist" % (self.base_path))
        elif not os.path.exists(self.method_path):
            try:
                os.mkdir(self.method_path)
            except Exception, e:
                print repr(e)
        self.mg_path = os.path.join(self.method_path, 'multi_goal')
        try:
            os.mkdir(self.mg_path)
        except Exception as e:
            print repr(e)
            
    def _prepare_RL_variables(self):
    
        action_outputs = 2
        nfq_action_outputs = 4
        self.RL = rl_methods.mgnfq.MGNFQ(os.path.join(self.lowest_path, "nfq/multi_goal"),
                                        nfq_action_outputs,
                                        maze_shape = (self.x_len * (1 / self.metric_resolution), self.y_len * (1 /self.metric_resolution))
                                        , **self.nfq_args)
        self.trial_length = 1000 #seconds
        self.trial_begin =rospy.Time().now().to_time()
        self.trial_time = lambda : rospy.Time().now().to_time() - self.trial_begin 
        
        
    def _prepare_next_init(self):
        rospy.loginfo('learned This Goal from the current Init position, Saving Analysis and changing Init position')
        
        self.init_index += 1
        if self.init_index >= self.init_no:
            self.init_index = 0
            self._prepare_next_goal()
        
        self.x_init, self.y_init = self.full_info[self.active_goal][1][self.init_index]
        
        self.base_path = os.path.join(self.lowest_path,'goal_'+str(self.active_goal)+'/')
        
        self._prepare_init_n_goal_position((self.goal_set[self.active_goal].x,self.goal_set[self.active_goal].y,0))
        self.__reset_performance_variables()
        self.consecutive_succes = list(numpy.zeros(15)) 
        
    def _prepare_next_goal(self):
        ''' Reset all RL data, and goes to the next goal.
        Exits if all goals are reached.
        '''
        rospy.loginfo('learned This Goal, Saving Analysis and changing active goal')
        self.goal_path = self.base_path
        self.save_results(self.consecutive_succes, self.actions_numbers)
        self.fig1_data=[]       
        self.fig2_data=[]
        
        self.active_goal = self.select_active_goal()
        self.consecutive_succes = list(numpy.zeros(15))
        
        #self.save_maze(self.goal_set[self.active_goal].x, self.goal_set[self.active_goal].y, self.goal_path, self.active_goal)   
        
        if self.active_goal == -1:
            print "Signaling Shutdown"
            #TODO: override del function to cleanly exit
            try:
                rospy.signal_shutdown("All goals are reached. Exiting Programs")
            except Exception as e:
                print "Clean shutdown Failed, forcing Python Exit"
                exit()
            '''
            if self.pose_index >= len(self.goals):
            print "Signaling Shutdown"
            self.__del__()
            '''
            
    def _process_odometry(self, data):
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
        self.radian = degrees[2]
        
    def get_state_representation(self, image, output = "location"):
        ''' Current pose + Goal pose arrays'''
        state = None
        if output == "autoencoder": #Only use the encoder part
            state =  self.encode(image)[0]
        elif output == "location": #Use the location estimations
            out = self.encode_n_estimate(image)[0]
            final = scale(out)
            #print "X:%s, Y:%s" % (final[0], final[1])
            state = out
        else:
            res = self._convert_to_grid(self.x, self.y, self.radian)
            state = res
        
        state , pos = state 
        state_list = [state] * len(self.goal_set)
            
        return state_list, pos
    
    def select_active_goal(self):
        '''Selects current goal based on TD information on all the goals and distances to the goal'''
        #TODO: Think about how to include environment information to select goals
        highest_td = float('-inf')
        next_goal_idx = -1
        for i in range(len(self.goal_set)):
            if self.reached_goals[i] == 1:
                rospy.loginfo('skipping reach goal No. %d', i)
                continue
            goal = self.goal_set[i]
            goal_td = numpy.mean(self.RL.td_map[i][goal.x_ind,goal.y_ind,:])
            rospy.loginfo("Goal number %d, has TD Error %d", i, goal_td)
            if goal_td > highest_td:
                highest_td = goal_td
                next_goal_idx = i
        
        if next_goal_idx != -1:
            rospy.loginfo("Next Goal is %d, with TD Error of %d", next_goal_idx, goal_td)
        else:
            rospy.loginfo("All goals traversed")
        return next_goal_idx
    #TODO: replace this callback with a simple function and use tf listener  
    def odomcb(self):
        
        global NUM1, NUM2, LOCK1, LOCK2, ARR1, ARR2
        
        #Switched from Callback to wait_for_message
        data = rospy.wait_for_message('/odom', Odometry)
        self._process_odometry(data)
        
        try:
            self.loop_rate.sleep()
        except ROSTimeMovedBackwardsException as e:
            print repr(e)
            
        if self.running and self.mg_running:
            self.steps_number += 1
            
            state_list, pos = self.get_state_representation(None, output = self.input_type)
            state= state_list[self.active_goal]    
            
            action = self.RL.select_action(state, self.active_goal)
            
            if self.method == "cacla":
                self._perform_action(action) #Gives absolute speed values
            else:
                self._perform_movebase_action_no_rot(action, pos)
                
            #Re updated the odometry after performing the action
            updated_pos = rospy.wait_for_message('/odom', Odometry)
            
            self.x = updated_pos.pose.pose.position.x
            self.y = updated_pos.pose.pose.position.y    
            next_state_list, pos = self.get_state_representation(None, output = self.input_type)
            next_state = next_state_list[self.active_goal]    
                    
            reward_list = self.get_reward()
            main_goal_reward = reward_list[self.active_goal]

            times_up = self.steps_number >= self.trial_max_steps
                
            ####Begin - ADD Data to Batch for later Update####
            if main_goal_reward >= self.angular_fix_reward: 
                #TODO: how to solve the reach goal for goal_set or traversing locations as goals?
                rospy.logerr("#########################################################################################")
                rospy.logerr("#########################################################################################")
                rospy.logerr("GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOAAAAAAAAAAAAAAAAAAAALLLLLLL REAAAAAAAAAAAAAAAAAAAAAACHED")
                rospy.logerr("#########################################################################################")
                rospy.logerr("#########################################################################################")
                self.success_trials += 1
                next_state_list[self.active_goal] = self.empty_vector
                #TODO: perhaps just only apply reward here? and forget next state list?
                self.RL.update_batch(state_list, action, reward_list, next_state_list, self.active_goal)
            elif times_up:
                self.failed_trials += 1
                self.RL.update_batch(state_list, action, reward_list, next_state_list, self.active_goal)
            else:
                self.RL.update_batch(state_list, action, reward_list, next_state_list, self.active_goal)
            ####End - ADD Data to Batch for later Update####
                 
            self.previous_action = action
            
            rospy.logdebug(" Reward: %3.2f, action %s" %( main_goal_reward ,action))
            
            ####Perform batch update
            if self.RL.total_actions > self.train_frequency_action:
                self._initiate_batch_update()

            if self.RL.total_actions > self.replay_frequency_action:
                self.replay_frequency_action += self.replay_frequency_step
                rospy.loginfo("EXPERIENCE__REPLAY")
                self.RL.update_networks_plus(experience_replay = True)
                rospy.loginfo("EXPERIENCE__REPLAY_FINISHED")
                #self.__reset_performance_variables()

                self.fig1_data.append([self.RL.total_actions, 
                                self.RL.cum_reward / self.train_frequency_step])

            #### Figure Updated #####  
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
            ###########################
            
            if times_up or main_goal_reward >= self.angular_fix_reward:
            #self.success_trials += 1
            #if True:
                self.actions_numbers.append(self.steps_number)
                if len(self.actions_numbers) > 200:
                    self.actions_numbers=self.actions_numbers[1:]

                
                self.fig2_data.append([self.RL.progress, self.RL.steps])

                success=float(self.success_trials)/float(self.success_trials + self.failed_trials)
                self.consecutive_succes.append(success)
                if len(self.consecutive_succes)>15:
                    self.consecutive_succes=self.consecutive_succes[1:]
                
                if (self.failed_trials == 0 and self.success_trials == 0):
                    pass #This happens after a batch update
                else:
                    print "SUCCESS RATIO: ", float(self.success_trials)/float(self.success_trials + self.failed_trials)
                    #pass
                self.running = False
                self.RL.save()
                
                self.steps_data.append((self.RL.progress, self.RL.steps))
                
                try:
                    self.total_reward_queue.put_nowait(self.RL.total_reward)
                except Queue.Full:
                    #self.analyze_experiment()
                    pass
                   

                if np.asarray(self.consecutive_succes).mean() >= 0.80 and self.steps_number < 100:
                #if True:
                    self.reached_goals[self.active_goal] = 1
                    self.save_and_exit() #doesnt really exit, just saves
                    self._prepare_next_init()

                print 'Current goal:', self.active_goal
                print 'consecutive successes: ', self.consecutive_succes
                print 'steps numbers:        ', self.steps_number
                self.reset()
                self.steps_number = 0
        
    def get_reward(self):
        reward_list = []
        
        if self.in_collision:
            self.in_collision = False
            return [self.obstacle_punishment] * (self.goal_set.__len__())
        
        counter = 0 #for debugging
        for goal in self.goal_set:

            reward = 0
            current_loc = Position(x= self.x, y= self.y, yaw = self.radian)
        
            d_location = current_loc - goal 
            d_angle = math.fabs(goal.yaw - current_loc.yaw)
            
            if counter == self.active_goal:
                rospy.loginfo("Location Diff: %f", d_location)
            
            #Give reward based on distance
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
                
            #Check location distance for reward
            if d_location <= self.goal_xy_threshold:
                reward += self.location_fix_reward
            else:
                reward += self.fix_punishment
            
            #If robot was in good location, check for angular precision
            if d_location <= self.goal_xy_threshold:
                if d_angle <= self.goal_theta_threshold:
                    reward = self.angular_fix_reward
                    rospy.logdebug("Reached a Goal")
                    if counter == self.active_goal:
                        rospy.logwarn("Reached Current Goal.")
                else:
                    pass
            
            reward_list.append(reward)
            counter += 1
            
        return reward_list
    
    def save_results(self, consecutive_succes, actions_numbers):
    
        fig1=np.asarray(self.fig1_data)
        fig2=np.asarray(self.fig2_data)
        succ=np.asarray(consecutive_succes)
        actions=np.asarray(actions_numbers)

        np.save(os.path.join(self.goal_path,'mg_fig1'), fig1)    
        np.save(os.path.join(self.goal_path,'mg_fig2'), fig2)
        np.save(os.path.join(self.goal_path,'mg_success'), succ)
        np.save(os.path.join(self.goal_path,'mg_actions'), actions)
        #np.save(path+'maze_with_goal', self.maze_to_show)
    

if __name__ == '__main__':

    #TODO: fix reading goals and plotting.
    reference_frame = 'odom'
    
    path = os.environ['BORG'] + '/ros/catkin_ws/src/cacla_nav/goals_starts/'
    #path = os.environ['HOME'] + '/ros/catkin_ws/src/cacla_nav/goals_starts/'
    goals = np.load(path+'goals.npy')
    starts = np.load(path+'starts.npy')
    maze = np.load(path+'approx.npy')
    indeces = np.load(path+'indeces.npy')
    
    print 'starting positions'
    print starts
    time.sleep(1)
    ###################################################
    # Here specify which goal you want to use
    inds=[0,1,2,3,4,5,6,7,8,9]
    ###################################################
    goal_list = []
    x_init_list = []
    y_init_list = []
    goal_list_idx = []
    last_index_list = []
    goal_list_for_parent = []
    ordered_index_list = []
    temp_init_list = []
    full_info = []
    for ind in inds:
    
        x,y=goals[ind]
        
        print 'goal is : ', (x,y)

        fold=path+'goal_'+str(ind)+'/'
        tmp=copy.deepcopy(maze)
        try:
            os.mkdir(fold)
        except:
            pass
        try:
            os.mkdir(fold+'nfq')
        except:
            pass
        ind1,ind2=indeces[ind]
        tmp[ind1,ind2]=2
        
        f3=plt.figure(3)
        plt.matshow(tmp, fignum=3)
        f3.savefig(fold+'maze.png')
        plt.close('all')
        
        goal_list.append((x,y, ind1, ind2))
        for ind2, (x_init, y_init) in enumerate(starts):

            #x_init, y_init = (-8.1,-2.1)
            goal_list_for_parent.append((x,y,0))
            x_init_list.append(x_init)
            y_init_list.append(y_init)
            temp_init_list.append((x_init, y_init))
            #last_index_list.append(ind2)
            ordered_index_list.append((ind, x_init, y_init))
        full_info.append(((x,y,0), temp_init_list))
        temp_init_list = []
    rospy.sleep(rospy.Duration(5))
    
    experiment = MultiGoalStage(path, reference_frame, x_init_list, y_init_list, 
                                goal_list, ordered_index_list, goal_list_for_parent, full_info)
    
    
    while not rospy.is_shutdown():
        try:
            if experiment.running and experiment.mg_running:
                experiment.odomcb()
            else:
                pass
        except rospy.ROSInterruptException as e:
            print repr(e)
            break
    os.system('killall xterm -9')
    


    
