#!/usr/bin/env python
import matplotlib.pyplot as plt
import rospy, roslib
import time, math
import cPickle, sys, copy
import math, numpy, cv2, random
# import sd_autoencoder
import numpy as np

# sys.modules['sd_autoencoder'] = sd_autoencoder
import rl_methods.mgnfq as mgnfq
import rl_methods

import theano
from theano import tensor as T, config, shared
from theano.tensor.shared_randomstreams import RandomStreams
from fake_test import FAKE_test
# theano.config.profile = True
# theano.config.profile_memory = True

from sets import Set

import os
import Queue
from multiprocessing import Process, Pipe
import shutil
from collections import deque
import dill
import threading
    
class Position(object):

    def __init__(self, x = 0, y = 0, z = 0, roll = 0, pitch = 0, yaw = 0, x_ind = 0, y_ind = 0, theta_ind = 0):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.x_ind = x_ind
        self.y_ind = y_ind
        self.theta_ind = theta_ind
        
    '''These functions are required for Set Operations'''

    def __eq__(self, other):
        ''' only x,y and yaw are computed'''
        return other and self.x == other.x and self.y == other.y and self.yaw == other.yaw

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        ''' only x,y and yaw are computed'''
        return hash((self.x, self.y, self.yaw))
    
    def __sub__(self, other):
        '''manhattan distance, subtraction overload'''
        return math.fabs(self.x - other.x) + math.fabs(self.y - other.y)
    
    def __repr__(self):
        return "x: %f, y: %f, radian: %f" % (self.x, self.y, self.yaw)
    
    def s_euclidean_d(self, other):
        math.pow(self.x - other.x, 2) + math.pow(self.y - other.y, 2)


class MultiGoal(FAKE_test):

    def __init__(self, base_path, parent_conn, maze_path, maze_ind, results_path, **kwargs):
        super(MultiGoal, self).__init__(base_path, maze_path, maze_ind, **kwargs)
        
        self.trial_max_steps = 10000
        self.max_step_number = 20
        self.last_avg_steps_number = 200
        self.avg_steps_lists = [self.last_avg_steps_number] * 50
        self.avg_steps_number_difference = 100
        self.average_step_number = 15
        self.nfq_args = {'epsilon':(0.1, 0.1) \
        , 'discount_factor':0.99, 'random_decay': 0.998\
        , 'explore_strategy':3, 'activation_function':1
        , 'learning_rate':1.0, 'epsilon':0.0, 'temperature':2.0}
        # self.trial_max_steps = 10000
        self._prepare_init_list(maze_path, 0, ind = 0)
        self.__prepare_RL_variables()
        
        self.current_maze = 0
        self.save_path = results_path
        # self.goal_set = Set()
        # self.goal_set.add(Position(x=self.goal['x'], y=self.goal['y'], yaw = self.goal['theta']))
        # self.goal_set.add(Position(x=9, y=9, yaw = self.goal['theta']))
        
        self.goal_set = []
        # self.goal_set.append(Position(x=self.goal['x'], y=self.goal['y'], yaw = self.goal['theta']))
        # self.goal_set.append(Position(x=9, y=9, yaw = self.goal['theta']))
        for x_g, y_g in self.multi_goals:
            self.goal_set.append(Position(x = x_g, y = y_g, yaw = 0))    
        
        # Which goal to follow! Default is 0 which is the first goal
        
        self.active_goal_list = []
        self.active_goal = 0
        self.active_goal_list.append(self.active_goal)
        self.maze_ind = maze_ind
        
        self.traversing_locs = Set()
        
        # Select below if goal and current state are part of input
        # ev_size = self.structure.size * 2
        # Select below if only current state is the input 
        ev_size = self.structure.size
        self.empty_vector = numpy.zeros((1, ev_size), dtype = theano.config.floatX) 
        self.empty_vector = numpy.reshape(self.empty_vector, -1)
        self.reached_goals = [0] * len(self.goal_set)
        
        # Used for plotting
        self.n_steps_list = []
        self.actions_numbers = []
        self.cum_reward_list = []
        
        self.consecutive_succes = list(numpy.zeros(15))
        self.consecutive_success_steps = list(numpy.ones(15) * self.max_step_number)
        self.fig1_data = []       
        self.fig2_data = []
        
        self.activation_function = T.nnet.relu
        
        plt.ion()
        
        self.save_timer = time.time()
        self.save_interval = 3600.0 #seconds
        
        self.avg_err = 100

    # def __easy_grid(self, x, y, theta, x_len = 8, y_len = 5):
    #    return super(MultiGoal, self).__easy_grid(self, x, y, theta, x_len = 8, y_len = 9)  
    def __reset_performance_variables(self):
        
        self.total_reward_queue = Queue.Queue(self.queue_size)
        self.success_ratio_threshold = 0.8  # The winning ratio required for updating the initial position
        self.success_trials = 0.0 
        self.failed_trials = 0.0
        self.last_position_update = 0
        self.win_ratio = 0   
    
    def __prepare_RL_variables(self):
        action_outputs = 2
        nfq_action_outputs = 4
        
        self.RL = rl_methods.mgnfq.MGNFQ(os.path.join(self.base_path, "nfq/multi_goal"),
                                         nfq_action_outputs, maze_shape = self.structure.shape,
                                         **self.nfq_args)
        
        
        self.trial_length = 10  # seconds
        self.trial_begin = time.time()
        # print self.trial_begin
        self.trial_time = lambda : time.time() - self.trial_begin 
        self.running = True            
    
    def _easy_grid(self, x, y, goal_x, goal_y):
        '''
        Converts the location and goal output to grid like format 
        '''

        rows, columns = self.structure.shape
        
        x_idx = x
        y_idx = y
        
        # rospy.logdebug( "x: %s, y: %s" % (x_idx,y_idx))
        pos_matrix = numpy.zeros((rows, columns), dtype = theano.config.floatX)
        pos_matrix[x, y] = 1.0
        pos_matrix = numpy.reshape(pos_matrix, -1)
        
        # If you want to have goal as input, uncomment here   
        '''
        goal_pos_matrix = numpy.zeros((rows, columns), dtype=theano.config.floatX)
        goal_pos_matrix[goal_x, goal_y] = 1.0
        goal_pos_matrix = numpy.reshape(goal_pos_matrix, -1) 
        #dot_product = numpy.dot(pos_matrix, goal_pos_matrix)
        
        h_stack = numpy.hstack((pos_matrix, goal_pos_matrix))
        '''
        return pos_matrix
        
        return h_stack   

    def reset(self):
        # TODO: below is only for gazebo map
        self.x = self.initx 
        self.y = self.inity 
        self.radian = self.initradian = 0.0

        self.in_collision = False
        
        self.RL.reset()
        self.trial_begin = time.time()
        self.running = True
        
    def select_active_goal(self):
        '''Selects current goal based on TD information on all the goals and distances to the goal'''
        # TODO: Think about how to include environment information to select goals
        highest_td = float('-inf')
        next_goal_idx = -1
        for i in range(len(self.goal_set)):
            if self.reached_goals[i] == 1:
                rospy.loginfo('skipping reached goal No. %d', i)
                continue
            goal = self.goal_set[i]
            goal_td = numpy.mean(self.RL.td_map[i][goal.x, goal.y, :])
            rospy.loginfo("Goal number %d, has TD Error %d", i, goal_td)
            if goal_td > highest_td:
                highest_td = goal_td
                next_goal_idx = i
        
        if next_goal_idx != -1:
            rospy.loginfo("Next Goal is %d, with TD Error of %d", next_goal_idx, goal_td)
        else:
            rospy.loginfo("All goals traversed")
        return next_goal_idx
    
    def runner_batch(self):
        
        to_save = []
           
        while True:
            if self.running:
                self.steps_number += 1
                terminal = False
                skip_update = False
                
                state_list = self.get_state_representation()         
                action = self.RL.select_action(state_list[self.active_goal], self.active_goal)
                self._perform_simple_action(action)
                next_state_list = self.get_state_representation()
                reward_list = self.get_reward()
                times_up = self.steps_number >= self.trial_max_steps
                
                main_goal_reward = reward_list[self.active_goal]

                ####Begin - ADD Data to Batch for later Update####
                if main_goal_reward >= self.angular_fix_reward: 
                    # TODO: how to solve the reach goal for goal_set or traversing locations as goals?
                    self.success_trials += 1
                    next_state_list[self.active_goal] = self.empty_vector
                    # TODO: perhaps just only apply reward here? and forget next state list?
                    self.RL.update_online(state_list, action, reward_list, next_state_list, self.active_goal, self.reached_goals)
                elif times_up:
                    self.failed_trials += 1
                    self.RL.update_online(state_list, action, reward_list, next_state_list, self.active_goal, self.reached_goals)
                else:
                    self.RL.update_online(state_list, action, reward_list, next_state_list, self.active_goal, self.reached_goals)
                ####End - ADD Data to Batch for later Update####
                
                ####Begin - Perform batch update ####
                if self.RL.total_actions > self.train_frequency_action:
                    self.cum_reward_list.append((self.RL.total_actions, self.RL.cum_reward / self.train_frequency_step))
                    # draw(1, self.cum_reward_list)
                    self.RL.cum_reward = 0

                    self.train_frequency_action += self.train_frequency_step
                    # print "Batch Update" 
                    # self.RL.update_networks_plus()
                ####End - Perform batch update ####    
                
                rospy.logdebug(" Reward: %3.2f     Action: %s", main_goal_reward, action)
                if times_up or main_goal_reward >= self.angular_fix_reward:
                    # print "Time: ", self.trial_time()
                    # print "Actions No: ", self.steps_number
                    self.actions_numbers.append(self.steps_number)
                    if len(self.actions_numbers) > 200:
                        self.actions_numbers = self.actions_numbers[1:]
                        
                    self.fig2_data.append([self.RL.progress, self.RL.steps])
                    
                    if (self.failed_trials == 0 and self.success_trials == 0):
                        pass  # This happens after a batch update
                    else:
                        print "SUCCESS RATIO: ", float(self.success_trials) / float(self.success_trials + self.failed_trials)
                        pass
                    self.running = False
                    
                    success = float(self.success_trials) / float(self.success_trials + self.failed_trials)
                    self.consecutive_succes.append(success)
                    self.consecutive_success_steps.append(self.steps_number)
                    to_save.append(success)
                    if len(self.consecutive_succes) > 50:
                        self.consecutive_succes = self.consecutive_succes[1:]
                    if len(self.consecutive_success_steps) > 50:
                        self.consecutive_success_steps = self.consecutive_success_steps[1:]
                        
                    self.RL.save()
                    self.save_state()
                    
                    
                    self.n_steps_list.append((self.RL.progress, self.RL.steps))
                    draw(3, self.n_steps_list)
                    # plt.figure(2)
                    # plt.scatter(self.RL.progress, self.RL.steps)
                    
                    try:
                        self.total_reward_queue.put_nowait(self.RL.total_reward)
                    except Queue.Full:
                        self.analyze_experiment()
                    
                    ###Begin - Experience Replay###
                    
                    if self.RL.total_actions > self.replay_frequency_action:
                        self.replay_frequency_action += self.replay_frequency_step
                        print "-----------Replay XP------------------"
                        self.avg_err = self.RL.update_networks_plus(experience_replay = True, reached_goals = self.reached_goals)
                    ###End - Experience Replay###            
                     
                    self.avg_steps_number_difference = math.fabs(self.last_avg_steps_number - 
                                                            numpy.asarray(self.consecutive_success_steps).min())
                    self.avg_steps_lists.pop(0)
                    self.avg_steps_lists.append(self.avg_steps_number_difference)
                    print "average step difference ", self.avg_steps_number_difference
                    print "min steps ", numpy.asarray(self.consecutive_success_steps).min()
                    print "avg steps mean ", numpy.asarray(self.avg_steps_lists).mean()
                    print "temperature: ", self.RL.temperature
                    if self.avg_err > 0:
                        print "latest total avg XP replay err", self.avg_err
                    if (numpy.asarray(self.consecutive_succes).mean() >= 0.999  
                        # and numpy.asarray(self.consecutive_success_steps).mean() < self.average_step_number  
                        # and numpy.asarray(self.consecutive_success_steps).min() < self.max_step_number):
                        #and numpy.asarray(self.avg_steps_lists).mean() <= 200
                        and self.avg_steps_number_difference < 10
                        ):
                        rospy.loginfo('learned This Goal, from Current Starting Point, Saving Analysis and changing active goal')
                        self.save_results(self.save_path, to_save, self.actions_numbers)
                        self.fig1_data = []       
                        self.fig2_data = []
                        to_save = []
                        self._prepare_next_init()
                        #self.RL.temperature = 2
                        self.avg_err = 1000
                        
                        
                    self.last_avg_steps_number = numpy.asarray(self.consecutive_success_steps).mean()    
                    print 'Current goal:', self.active_goal
                    print 'consecutive successes: ', self.consecutive_succes
                    print 'steps numbers:        ', self.steps_number
                    self.steps_number = 0
                    self.reset()

    def get_state_representation(self):
        ''' Current pose + Goal pose arrays'''
        state_list = []
        
        for goal in self.goal_set:
            combined_loc = self._easy_grid(self.x, self.y , goal.x, goal.y)
            state_list.append(combined_loc)

        for goal in self.traversing_locs:
            combined_loc = self._easy_grid(self.x, self.y , goal.x, goal.y)
            state_list.append(combined_loc)
            
        return state_list
        
    def save_results(self, path, consecutive_succes, actions_numbers):
    
        path = path + 'Maze_' + str(self.maze_ind) + '_' + str(self.current_maze) + '/'
        
        try:
            os.mkdir(path)
        except:
            print 'Existing folder!'

        fig1 = np.asarray(self.fig1_data)
        fig2 = np.asarray(self.fig2_data)
        succ = np.asarray(consecutive_succes)
        actions = np.asarray(actions_numbers)

        np.save(path + 'fig1', fig1)    
        np.save(path + 'fig2', fig2)
        np.save(path + 'success', succ)
        np.save(path + 'actions', actions)
        # np.save(path+'maze_with_goal', self.maze_to_show)
        self.current_maze += 1
        
    def save_state(self, force = False):
        if (time.time() - self.save_timer > self.save_interval) or force:
            base_path = os.path.join(self.base_path)
            filepath = os.path.join(base_path, 'maze_multi_goal_caferoom_state')
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
        filepath = os.path.join(base_path, 'maze_multi_goal_caferoom_state')
        
        try:
            f = file(filepath, 'rb')
            self.__dict__.update(dill.load(f))
        except Exception as ex:
            print "Loading of states went wrong"
            print ex
    
        
    def get_reward(self):
        reward_list = []
        
        if self.in_collision:
            self.in_collision = False
            return [self.obstacle_punishment] * (self.goal_set.__len__() + self.traversing_locs.__len__())
        
        counter = 0  # for debugging
        for goal in self.goal_set:

            reward = 0
            current_loc = Position(x = self.x, y = self.y, yaw = self.radian)
        
            d_location = current_loc - goal 
            
            d_angle = math.fabs(goal.yaw - current_loc.yaw)
            if counter == self.active_goal:
                rospy.logdebug("Goal Loc : " + repr(goal))
                rospy.logdebug("Location Diff: %d", d_location)
                rospy.logdebug("Active Goal: %d", self.active_goal)
            
            # Check location distance for reward
            if d_location <= self.goal_xy_threshold:
                reward += self.location_fix_reward
            
            else:
                reward += self.fix_punishment
            
            # If robot was in good location, check for angular precision
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
    
    def save_and_exit(self):
        self.base_path
        base_path = os.path.join(self.base_path, foldername)
        
        dirname = 'net%d' % self.active_goal
        filepath = os.path.join(base_path, dirname, self.network_filename)
            
        fig1 = np.asarray(self.fig1_data)
        fig2 = np.asarray(self.fig2_data)
        succ = np.asarray(self.consecutive_succes)
        actions = np.asarray(self.actions_numbers)

        np.save(self.base_path + 'fig1', fig1)    
        np.save(self.base_path + 'fig2', fig2)
        np.save(self.base_path + 'success', succ)
        np.save(self.base_path + 'actions', actions)
        
        rospy.signal_shutdown('done')
        
    def _prepare_init_list(self, path, num, ind = 0):
        self.init_index = 0
        self.starts = np.load(path + 'starts_' + str(num) + '.npy')
        self.init_no = self.starts.shape[0]

        (x_start, y_start) = self.starts[ind]

        self.x = self.initx = x_start
        self.y = self.inity = y_start
        self.radian = self.initradian = 0.0
    
    def _prepare_next_init(self):
        self.init_index += 1
        
        
        if self.init_index >= self.init_no:
            self.init_index = 0
            self.reached_goals[self.active_goal] = 1
            self.active_goal = self.select_active_goal()
            
            
            if self.active_goal == -1:
                # rospy.signal_shutdown("All goals are reached")
                self.save_state(force = True)
                raise KeyboardInterrupt
            self.active_goal_list.append(self.active_goal)
            self.RL.clear_memory()                  
        self.initx, self.inity = self.starts[self.init_index]
        
        # self.base_path = os.path.join(self.lowest_path,'goal_'+str(self.active_goal)+'/')
        # self._prepare_init_n_goal_position((self.goal_set[self.active_goal].x,self.goal_set[self.active_goal].y,0))
        self.__reset_performance_variables()
        self.consecutive_succes = list(numpy.zeros(15))
        self.consecutive_success_steps = list(numpy.ones(15) * self.max_step_number)
        self.avg_steps_number_difference = 100


def draw(fig_num, list):
    return
    plt.figure(fig_num)
    x_axis, y_axis = zip(*list)

    plt.scatter(x_axis, y_axis)     
    
    plt.pause(0.05)

    
def mp_plot(conn):
    # Endless loop to receive and draw stuff
    while True:
        obj = conn.recv()
        try:
            fig_num, list = obj
            print list
            x_axis, y_axis = zip(*list)
        except Exception as e:
            print repr(e)
            continue
        fig = plt.figure()   
        plt.plot(x_axis, y_axis)
        plt.ion()
        plt.show(block = False)
        plt.close(fig)
        plt.close('all')

'''
Multi goal is supposed to solve multiple mazes using the MGNFQ approach. This file is for the big cafe room
It requires a path to read the Maze_dataset, and a valid path to dump results.
#############################################
The Multi goal reads the maze structure with the init states and the goals from a folder (Currently Maze_dataset)
It then tries to reach each goal from all the initial positions.

For each Goal to be accepted as successfull, it should first Have at least X=15 consecutive wins. Then, the average
number of actions towards goals should be lower than Y=200, and finally, the minimum number of axtions should be lower than
Z=100

The results for each goal is stored in Multy_Maze_results folder.
To depict the Figures, use DataAnalisi_Maze
#############################################

'''
        
def timer():
    global save_now
    save_now = True;
        
if __name__ == '__main__':
    save_now = False
    parent_conn, child_conn = Pipe()
    # p = Process(target=mp_plot, args=(child_conn,))
    # p.start()
    
    base_path = os.path.join(os.environ['HOME'], "amir-nav-experiments/maze_files/caffe_room") 
    
    maze_path = base_path + '/Maze_dataset/'
    results_path = base_path + "/Multy_Maze_results/"
    maze_ind = 0
    maze_number = 1
    # maze_path = None
    goal_order_list = []
    for i in xrange(maze_ind, maze_number):
        maze_ind = i
        experiment = MultiGoal(base_path, parent_conn, maze_path, maze_ind, results_path)
        experiment.load_state()
        try:
            experiment.runner_batch()
            
        except KeyboardInterrupt:
            experiment.save_state(force = True)
            goal_order_list.append(experiment.active_goal_list)
            print "Maze %d done" % i
          
        experiment.reset()
        del experiment
    a = np.asarray(goal_order_list)
    np.save(maze_path + '/goal_order.npy', a)
    rospy.signal_shutdown("All goals are reached")

    # p.join()

