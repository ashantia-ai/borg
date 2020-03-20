#!/usr/bin/env python
#import sd_autoencoder
#sys.modules['sd_autoencoder'] = sd_autoencoder

import matplotlib.pyplot as plt
import rospy, roslib
import time, math
import cPickle, sys, copy
import math, numpy, cv2, random
import numpy as np


import rl_methods.nfq as nfq
import rl_methods

import theano
from theano import tensor as T, config, shared
from theano.tensor.shared_randomstreams import RandomStreams

#theano.config.profile = True
#theano.config.profile_memory = True

import os
import Queue

import dill

from multiprocessing import Process, Pipe
    
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


class FAKE_test(object):
    def __init__(self, base_path, maze_path=None, maze_ind=0, result_path = None, goal_ind = 0, **kwargs):
        
        self.steps_number = 0
        
        self.trial_max_steps = 2000
        self.max_step_number = 20
        self.last_avg_steps_number = 200
        self.avg_steps_lists = [self.last_avg_steps_number] * 50
        self.avg_steps_number_difference = 100
        self.average_step_number = 15
        self.nfq_args = {'epsilon':(0.1, 0.1) \
        , 'discount_factor':0.99, 'random_decay': 0.998\
        , 'explore_strategy':3, 'activation_function':1
        , 'learning_rate':1.0, 'epsilon':0.0, 'temperature':2.0, 'max_replay_size':60000}
        
        
        
        self.current_maze = goal_ind * 10
        self.save_path = result_path
        
        self.active_goal_list = []
        self.active_goal = goal_ind
        self.active_goal_list.append(self.active_goal)
        self.maze_ind = maze_ind
        self.reached_goals = [0] * 10
        
        if maze_path == None:
            self.random_maze = True
        else:
            self.random_maze = False
            
        self.maze_path = maze_path
        self.maze_ind = maze_ind

        self.__init_ros_node()
        self.__prepare_file_structure(base_path)
        
        self.input_type = "ground_truth" #Options: 1- autoencoder . 2-location, 3- ground truth location, Default: ground truth
        self.method = "nfq" #1- Cacla 2- NFQ
        
        self.__prepare_init_n_goal_position()
        self._prepare_init_list(maze_path, 0, ind = 0)
        self.__prepare_rewards()
        
        self.__prepare_action_variables()
        
        self.__prepare_performance_variables()
        self.__prepare_experience_replay()
        
        self.goal_set = []
        # self.goal_set.append(Position(x=self.goal['x'], y=self.goal['y'], yaw = self.goal['theta']))
        # self.goal_set.append(Position(x=9, y=9, yaw = self.goal['theta']))
        for x_g, y_g in self.multi_goals:
            self.goal_set.append(Position(x = x_g, y = y_g, yaw = 0)) 
        
        if kwargs:
            self.set_parameters(**kwargs)
        
        self.__prepare_RL_variables()
        
         # Used for plotting
        self.n_steps_list = []
        self.actions_numbers = []
        self.cum_reward_list = []
        
        self.consecutive_succes = list(numpy.zeros(15))
        self.consecutive_success_steps = list(numpy.ones(15) * self.max_step_number)
        self.fig1_data = []       
        self.fig2_data = []
        
        self.activation_function = T.nnet.relu
        self.save_timer = time.time()
        self.save_interval = 3600.0 #seconds
        
        self.avg_err = 100
        #self.reset()

    def __repr__(self, *args, **kwargs):
        return object.__repr__(self, *args, **kwargs)
    
    def _easy_grid(self, x, y, theta, x_len=9, y_len=9):
        '''
        Converts the location output to grid like format
        '''

        rows, columns = self.structure.shape
        depth = 1 #2 * Pi / 1.57
        
        x_idx = x
        y_idx = y
        
        rospy.logdebug( "x: %s, y: %s" % (x_idx,y_idx))
        pos_matrix = numpy.zeros((rows, columns), dtype=theano.config.floatX)
        pos_matrix[x, y] = 1.0
        
        return numpy.reshape(pos_matrix, -1)
    
    def _convert_to_grid(self, x, y, theta, x_len = 8, y_len = 5):
        '''
        Converts the location output to grid like format
        '''
        angular_resolution = 1.57
        rows, columns = self.structure.shape
        depth = 4 #2 * Pi / 1.57
        
        x_idx = self.x
        y_idx = self.y
        
        d_idx = int(theta / angular_resolution)
        rospy.logdebug( "x: %s, y: %s, theta %s" % (x_idx,y_idx, d_idx))
        pos_matrix = numpy.zeros((columns, rows, depth), dtype=theano.config.floatX)
        pos_matrix[x_idx, y_idx, d_idx] = 1.0
        
        return numpy.reshape(pos_matrix, -1)
              
    def __create_directories(self):
        if not os.path.exists(self.base_path):
            raise Exception("Base path %s doesn't exist" % (self.base_path))
        elif os.path.exists(os.path.join(self.base_path, "cacla")):
            try:
                os.mkdir(os.path.join(self.base_path, "cacla"))
            except Exception, e:
                print repr(e)
                
            
        
    
    def __init_ros_node(self):
        ''' Initialized the ros node and the main publishers
        '''
        rospy.init_node('Fake_nav')
        rospy.loginfo("Node initialized")

          
    def __load_network(self):
        try:
            f = file(self.theano_file, 'rb')
            model = cPickle.load(f)
            f.close()
            return model
        except Exception, e:
            raise Exception("Error in loading network: %s " % (repr(e))) 
        
    def _perform_simple_action(self, action_num):
        x = self.x
        y = self.y
        
        if self.structure[self.x, self.y] == 1:
            rospy.logerr("INSIDE OBJECTS.")
        theta = self.radian
        
        rospy.logdebug("Current Location is X: %s, Y: %s, T: %s. Action is : %s " % (x, y, theta, action_num))
        if action_num == 0:
            x -= 1
        elif action_num == 1:
            y -= 1
        elif action_num == 2:
            x += 1
        elif action_num == 3:
            y += 1
            
        x_check = lambda f: f < 0 or f >= (self.structure.shape[0])
        y_check = lambda f: f < 0 or f >= (self.structure.shape[1])
        if x_check(x) or y_check(y) or self.structure[x, y] == 1:
            self.in_collision = True
            rospy.logdebug("Destination Location is X: %s, Y: %s, T: %s COLLISION with Action %s" % (x, y, theta, action_num))
            return False
        
        self.x = x
        self.y = y
        self.radian = theta
        
        rospy.logdebug("Destination Location is X: %s, Y: %s, T: %s" % (x, y, theta))
        
        return True
        
    def _perform_discreet_action(self, action_num):
        #Moves the robot through the maze
        x = self.x
        y = self.y
        theta = self.radian
        epsilon = 0.1
        
        rospy.logdebug("Current Location is X: %s, Y: %s, T: %s. Action is : %s " % (x, y, theta, action_num))
        if action_num == 0:
            if math.fabs(theta - 0) < epsilon:
                x -= 1
            elif math.fabs(theta - 1.57) < epsilon:
                y -= 1
            elif math.fabs(theta - 3.14) < epsilon:
                x += 1
            elif math.fabs(theta - 4.71) < epsilon:
                y += 1
            else:
                raise
        
        elif action_num == 1:
            if math.fabs(theta - 0) < epsilon:
                x += 1
            elif math.fabs(theta - 1.57) < epsilon:
                y += 1
            elif math.fabs(theta - 3.14) < epsilon:
                x -= 1
            elif math.fabs(theta - 4.71) < epsilon:
                y -= 1
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
                
        elif action_num == 4:
            pass
        
        x_check = lambda f: f < 0 or f >= self.structure.shape[0]
        y_check = lambda f: f < 0 or f >= self.structure.shape[1] 
        if x_check(x) or y_check(y) or self.structure[x, y] == 1:
            self.in_collision = True
            rospy.logdebug("Destination Location is X: %s, Y: %s, T: %s COLLISION with Action %s" % (x, y, theta, action_num))
            return False
        
        self.x = x
        self.y = y
        self.radian = theta
        
        rospy.logdebug("Destination Location is X: %s, Y: %s, T: %s" % (x, y, theta))
        
        return True
    
    def __prepare_action_variables(self):
        #TODO: needs to called from set_params too
        #Action Variables
        self.in_collision = False
        if self.method == "cacla":
            self.previous_action = [0,0]
        else:
            self.previous_action = 0
            
    def __prepare_experience_replay(self):
        #Experience Replay
        self.state_history = []
        self.success_history = []
        self.replay_frequency = 100
        self.train_frequency = 10
        self.train_frequency_action = 1000
        self.train_frequency_step = 1000
        self.replay_frequency_action = 5000
        self.replay_frequency_step = 5000
               
    def __prepare_file_structure(self,base_path):

        self.base_path = base_path
        self.__create_directories()


    def __generate_random_maze(self, shape=(10,10), wall_freq=0.3):

        empty=numpy.zeros(shape)
        max_vertical_wall_lengh = int(shape[0]/2)
        max_orizontal_wall_lengh = int(shape[1]/2)

        for ind,raw in enumerate(empty):

            make_wall=random.random()
            if make_wall>(1-wall_freq):

                wall_lengh=random.randint(1,max_orizontal_wall_lengh)
                wall_start=random.randint(0,shape[1])
                if wall_start+wall_lengh > shape[1]:
                    raw[wall_start:]=1
                else:
                    raw[wall_start:wall_start+wall_lengh]=1
                empty[ind]=raw        

        for column_ind in xrange(len(empty[0])):

            make_wall=random.random()
            if make_wall>(1-wall_freq):

                column=empty[:,column_ind]
                wall_lengh=random.randint(1,max_orizontal_wall_lengh)
                wall_start=random.randint(0,shape[0])
                if wall_start+wall_lengh > shape[0]:
                    column[wall_start:]=1
                else:
                    column[wall_start:wall_start+wall_lengh]=1
                empty[:,column_ind]=column

        maze=self.__check_closed_room_in_maze(empty)
    
        return maze


    def __check_closed_room_in_maze(self, maze):
        tmp_maze=copy.deepcopy(maze)
        fin=copy.deepcopy(maze)
        tmp_maze[0,0]=2
        done=False
        while not done:
            changed=0
            for ind,raw in enumerate(tmp_maze):
                for ind2 in xrange(len(raw)):
                    if tmp_maze[ind, ind2]==2:

                        if ind != 0:
                            if tmp_maze[ind-1, ind2]==0:
                                tmp_maze[ind-1, ind2]=2
                                changed+=1

                        if ind != len(tmp_maze)-1: 
                            if tmp_maze[ind+1, ind2]==0:
                                tmp_maze[ind+1, ind2]=2
                                changed+=1

                        if ind2 != 0:
                            if tmp_maze[ind, ind2-1]==0:
                                tmp_maze[ind, ind2-1]=2
                                changed+=1

                        if ind2 != len(raw)-1:
                            if tmp_maze[ind, ind2+1]==0:
                                tmp_maze[ind, ind2+1]=2
                                changed+=1

            if changed==0: done=True
        ind_to_block=tmp_maze[:,:]==0
        fin[ind_to_block]=1
        return fin                        


    def __visualize_maze(self, maze):
        ##plt.figure()
        #plt.matshow(maze)
        #plt.show()
        pass
                    
    def __chose_random_goal_with_min_dist(self, maze, min_dist=7):
        
        new=numpy.zeros(maze.shape)
        to_show=copy.deepcopy(maze)
        tot_goals=0
        goals=[]
        for ind, row in enumerate(maze):
            for ind2 in xrange(len(row)):
                if ind+ind2>min_dist:
                    if maze[ind, ind2]==0:
                        new[ind,ind2]=1
                        tot_goals+=1
                        goals.append((ind,ind2))

        if tot_goals==0:
            print 'This maze is too small, please create another'
            return (to_show, None, None)
        random_goal_ind=random.randint(0, tot_goals-1)
        (x,y)=goals[random_goal_ind]
        to_show[x,y]=2

        return (to_show, x, y)
        
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
            print 'learned Goal: ', self.active_goal
            raise KeyboardInterrupt
            self.init_index = 0
            self.reached_goals[self.active_goal] = 1
            self.active_goal = self.active_goal + 1
            #self.RL.reset_temperature()
            self.RL.temperature = 2.0
            
            #if self.active_goal == -1 or self.active_goal > 9:
            #    # rospy.signal_shutdown("All goals are reached")
            #    self.save_state(force = True)
            #    raise KeyboardInterrupt
            self.active_goal_list.append(self.active_goal)
            self.RL.clear_memory()                  
        self.initx, self.inity = self.starts[self.init_index]
        
        # self.base_path = os.path.join(self.lowest_path,'goal_'+str(self.active_goal)+'/')
        # self._prepare_init_n_goal_position((self.goal_set[self.active_goal].x,self.goal_set[self.active_goal].y,0))
        self.__reset_performance_variables()
        self.consecutive_succes = list(numpy.zeros(15))
        self.consecutive_success_steps = list(numpy.ones(15) * self.max_step_number)
        self.avg_steps_number_difference = 100

        
    def __prepare_init_n_goal_position(self):
        if self.random_maze:
            while True:
                self.structure = self.__generate_random_maze()
                to_show, x_goal, y_goal=self.__chose_random_goal_with_min_dist(self.structure)
                self.__visualize_maze(to_show)
                #TODO: Temporary, remove it
                numpy.save("/home/ashantia/maze.npy",self.structure)
                a=raw_input('press 1 to recreate random maze, 2 to continue, 3 to exit \n')
                if a=='2': break
                elif a=='3': sys.exit()
        #############################################################################
        # This part is needed only when multi goal approach is used. External class #
        #############################################################################
        else:
            self.structure = numpy.load(self.maze_path+'maze_'+str(self.maze_ind)+'.npy')
            self.multi_goals =  numpy.load(self.maze_path+'goal_'+str(self.maze_ind)+'.npy')
            to_show = copy.deepcopy(self.structure)
            for x_goal, y_goal in self.multi_goals:
                #x_goal, y_goal = self.multi_goals[0]
                if to_show[x_goal, y_goal] != 0: print 'Error here'
                to_show[x_goal, y_goal] = 2
            self.__visualize_maze(to_show) 
        ################################################################
        self.empty_vector = numpy.zeros((1, self.structure.size))
        self.empty_vector = numpy.reshape(self.empty_vector, -1)
        self.x = self.initx = 1
        self.y = self.inity = 1
        self.radian = self.initradian = 0.0
        
        #Initial Position update after a successfull learning
        self.yaw_change = 90 #in degrees
        self.x_change = 50 #in cm
        self.y_change = 50 #in cm
        
        #Goal Parameters
        self.goal_list = []
        
        #self.goal = {'x':3.66, 'y':-0.59, 'theta':-90} #Next to stove, facing to it
        self.goal = {'x':x_goal, 'y':y_goal, 'theta':0.0} # infront of init position.
        
        self.goal_theta_threshold = 0.1 #Degrees
        self.goal_xy_threshold = 0.1 #Meters

    
    def __prepare_performance_variables(self):
        
        self.queue_size = 100
        self.total_reward_queue = Queue.Queue(self.queue_size)
        
        self.acceptable_run_threshold = 50 #Average total reward of 100 runs should be bigger than 50
        self.success_trials_threshold = 10
        self.success_ratio_threshold = 0.8 #The winning ratio required for updating the initial position
        self.success_trials = 0.0 
        self.failed_trials = 0.0
        self.last_position_update = 0
        self.fail_trial_threshold = 10000 #Number of failed trials before sigma is resetted to a higher value
        self.min_epsilon_update = 0.3 #Only reset epsilon if it is smaller than this value
        self.win_ratio = 0
        
    def __prepare_rewards(self):
        #The reward if location of the robot is close to goal
        self.location_fix_reward = 0.5
        #The reward if also the angle is correct
        self.angular_fix_reward = 100.0
        
        #Negative reward per time step
        self.fix_punishment = -0.1
        #Punishment for not reaching the goal
        self.trial_fail_punishment = -0.1
        #Punishmend for hitting an obstacle
        self.obstacle_punishment = -2.0
        self.negative_speed_punishment = -0.0
        
        
    def __prepare_RL_variables(self):
        action_outputs = 2
        nfq_action_outputs = 4
          
        
        self.RL = rl_methods.nfq.NFQ(os.path.join(self.base_path, "nfq"), nfq_action_outputs, **self.nfq_args)
        
        self.trial_length = 10 #seconds
        self.trial_begin = time.time()
        #print self.trial_begin
        self.trial_time = lambda : time.time() - self.trial_begin 
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
        f = open(os.path.join(self.base_path, "nfq", "analysis.txt"),'a')
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
    
                if self.RL.progress - self.last_position_update > self.fail_trial_threshold:
                    if self.RL.epsilon < self.min_epsilon_update:
                        #self.RL.epsilon = self.cacla_args['epsilon'][1] #Resetting Sigma to high value again
                        report += "*******epsilon IS RESETED*********************\n"
                     
                break
        f.write(report)
        f.close()
        
    def get_current_status(self, score = 0):
        status =  "------ Iteration No %s ------\n" % (self.RL.progress)
        status += "Start Position X: %s Y: %s Theta: %s\n" % (1, 
                                                              1, 0)
        status += "Goal Position X: %s Y: %s Theta: %s\n" % (self.goal['x'], self.goal['y'], self.goal['theta'])
        status += "Average reward over last %s runs: %s\n\n" % (self.queue_size, score)
        status += "Changing Parameters Section:\n"
        status += "    epsilon: %f\n" % (self.RL.epsilon)
        
        return status
    
           
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
        
        self.x = self.initx 
        self.y = self.inity 
        self.radian = self.initradian = 0.0

        self.in_collision = False
        
        self.RL.reset()
        self.trial_begin = time.time()
        self.running = True
        
    def runner_online(self):
        to_save = []
        if self.running:
            self.steps_number += 1
            terminal = False
            skip_update = False
            
            state = self.get_state_representation()         
            action = self.RL.select_action(state)
            self._perform_simple_action(action)
            next_state = self.get_state_representation()
            reward = self.get_reward()
            
            times_up = self.steps_number >= self.trial_max_steps
            
            if reward >= self.angular_fix_reward: 
                self.success_trials += 1
                self.RL.update_online(state, action, reward, self.empty_vector)
            elif times_up:
                self.failed_trials += 1
                self.RL.update_online(state, action, reward, next_state)
            else:
                self.RL.update_online(state, action, reward, next_state)
                
            if self.RL.total_actions > self.train_frequency_action:
                    
                #plt.figure(1)
                #plt.scatter(self.RL.total_actions, self.RL.cum_reward / self.train_frequency_step)
                self.RL.cum_reward= 0
                #plt.draw()
                    
                self.train_frequency_action += self.train_frequency_step
                
            rospy.logdebug(" Reward: %3.2f     Action: %s", reward, action)
            if times_up or reward >= self.angular_fix_reward:
                #self.steps_number = 0
                
                #Append the current steps number to the list, and keep the last 200 trials
                self.actions_numbers.append(self.steps_number)
                if len(self.actions_numbers) > 200:
                    self.actions_numbers = self.actions_numbers[1:]
                    
                self.fig2_data.append([self.RL.progress, self.RL.steps])
                
                if (self.failed_trials == 0 and self.success_trials == 0):
                    pass #This happens after a batch update
                else:
                    #print "SUCCESS RATIO: ", float(self.success_trials) / float(self.success_trials + self.failed_trials)
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
                    
                #Save state of RL. and check whether we should do a program snap shot
                self.RL.save()
                #self.save_state()
                
                #Add total progress and steps to the n_steps_list
                self.n_steps_list.append((self.RL.progress, self.RL.steps))
                draw(3, self.n_steps_list)
                
                try:
                    self.total_reward_queue.put_nowait(self.RL.total_reward)
                except Queue.Full:
                    self.analyze_experiment()
                
                #Initiate a experience replay if the number of actions reached the required frequency
                if self.RL.total_actions > self.replay_frequency_action:
                    self.replay_frequency_action += self.replay_frequency_step
                    print "-----------Replay XP------------------"
                    self.avg_err = self.RL.update_networks_plus(experience_replay = True)
                
                
                #Check average step numbers for this goal
                self.avg_steps_number_difference = math.fabs(self.last_avg_steps_number - 
                                                            numpy.asarray(self.consecutive_success_steps).min())
                self.avg_steps_lists.pop(0)
                self.avg_steps_lists.append(self.avg_steps_number_difference)
                print "average step difference ", self.avg_steps_number_difference
                print "min steps ", numpy.asarray(self.consecutive_success_steps).min()
                print "avg steps mean ", numpy.asarray(self.avg_steps_lists).mean()
                
                #Print average experience replay
                if self.avg_err > 0:
                    print "latest total avg XP replay err", self.avg_err
                    pass
                    
                #Goal convergance from current initial starting point criteria
                if (numpy.asarray(self.consecutive_succes).mean() >= 0.999  
                        #and numpy.asarray(self.avg_steps_lists).mean() <= 200
                        and self.avg_steps_number_difference < 10
                        ):
                    rospy.loginfo('learned This Goal from Current Starting Point, Saving Analysis and changing init pose')
                    self.save_results(self.save_path, to_save, self.actions_numbers)
                    self.fig1_data = []       
                    self.fig2_data = []
                    to_save = []
                    self._prepare_next_init()
                    self.avg_err = 100
                    
                self.last_avg_steps_number = numpy.asarray(self.consecutive_success_steps).mean()    
                print 'Current goal:', self.active_goal
                print 'consecutive successes: ', self.consecutive_succes
                print 'steps numbers:        ', self.steps_number
                self.steps_number = 0
                self.reset()
                
                
                
                #plt.figure(2)
                #plt.scatter(self.RL.progress, self.RL.steps)
                #plt.draw()
                
    def save_state(self, force = False):
        if (time.time() - self.save_timer > self.save_interval) or force:
            base_path = os.path.join(self.base_path)
            filepath = os.path.join(base_path, 'maze_single_goal_caferoom_state')
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
        filepath = os.path.join(base_path, 'maze_single_goal_caferoom_state')
        
        try:
            f = file(filepath, 'rb')
            self.__dict__.update(dill.load(f))
        except Exception as ex:
            print "Loading of states went wrong"
            print ex
            
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

    def runner_batch(self):
        if self.running:
            self.steps_number += 1
            terminal = False
            skip_update = False
            
            state = self.get_state_representation()         
            action = self.RL.select_action(state)
            self._perform_simple_action(action)
            next_state = self.get_state_representation()
            reward = self.get_reward()
            
            times_up = self.steps_number >= self.trial_max_steps
            
            if reward >= self.angular_fix_reward: 
                self.success_trials += 1
                self.RL.update_batch(state, action, reward, self.empty_vector)
            elif times_up:
                self.failed_trials += 1
                self.RL.update_batch(state, action, reward, next_state)
            else:
                self.RL.update_batch(state, action, reward, next_state)
            
            if self.RL.total_actions > self.train_frequency_action:
                self.RL.cum_reward= 0
                draw(1, self.RL.total_actions, self.RL.cum_reward / self.train_frequency_step)
                    
                self.train_frequency_action += self.train_frequency_step 
                self.RL.update_networks_plus()
                
            #if self.RL.total_actions > 10000:
            #    exit()
            
            rospy.logdebug(" Reward: %3.2f     Action: %s", reward, action)
            if times_up or reward >= self.angular_fix_reward:
                #print "Time: ", self.trial_time()
                #print "Actions No: ", self.steps_number
                self.steps_number = 0
                
                if (self.failed_trials == 0 and self.success_trials == 0):
                    pass #This happens after a batch update
                else:
                    #print "SUCCESS RATIO: ", float(self.success_trials)/float(self.success_trials + self.failed_trials)
                    pass
                self.running = False
                self.RL.save()
                
                #plt.figure(2)
                #plt.scatter(self.RL.progress, self.RL.steps)
                #plt.draw()
                
                try:
                    self.total_reward_queue.put_nowait(self.RL.total_reward)
                except Queue.Full:
                    self.analyze_experiment()
                                    
                

                if self.RL.total_actions > self.replay_frequency_action:
                    self.replay_frequency_action += self.replay_frequency_step
                    print "-----------Replay XP------------------"
                    self.RL.update_networks_plus(experience_replay = True)
                    
                self.reset()
                
    def __reset_performance_variables(self):
        
        self.total_reward_queue = Queue.Queue(self.queue_size)
        self.success_ratio_threshold = 0.8  # The winning ratio required for updating the initial position
        self.success_trials = 0.0 
        self.failed_trials = 0.0
        self.last_position_update = 0
        self.win_ratio = 0   
            
    def get_state_representation(self):
        #res = self._convert_to_grid(self.x, self.y, self.radian)
        res = self._easy_grid(self.x, self.y, self.radian)
            
        return res
        
    def get_reward(self):
        reward = 0
        
        if self.in_collision:
            self.in_collision = False
            return self.obstacle_punishment
        
        goal = self.goal_set[self.active_goal]
        #Get current location difference of the goal
        #d_location = ((abs(self.goal['x'] - self.x)) + (abs(self.goal['y'] - self.y)) ) 
        #d_angle = math.fabs(self.goal['theta'] - self.radian)
        
        current_loc = Position(x = self.x, y = self.y, yaw = self.radian)
        
        d_location = current_loc - goal 
        d_angle = math.fabs(goal.yaw - current_loc.yaw)
        
        #Check location distance for reward
        if d_location <= self.goal_xy_threshold:
            reward += self.location_fix_reward
        
        else:
            return reward + self.fix_punishment
        
        #If robot was in good location, check for angular precision
        if d_location <= self.goal_xy_threshold:
            if d_angle <= self.goal_theta_threshold:
                #TODO: Added Stopping Criteria for final reward. Has Magic Numbers. Fix it
                reward = self.angular_fix_reward
                rospy.logdebug("Reach Goal")
            else:
                pass #We pass here since currently we are not rotating the robot in the maze
                #return reward
        return reward
    
def draw(fig_num, list):
    return
    #plt.figure(fig_num)
    x_axis, y_axis = zip(*list)

    #plt.scatter(x_axis, y_axis)     
    
    #plt.pause(0.05)
    
def mp_draw(conn):
    #Endless loop to receive and draw stuff
    while True:
        pass
    
'''
This file is for single goal Runs. Please change the path for the correct Data set folder (kitchen or cafe room)
'''
        
if __name__ == '__main__':
    
    base_path = os.path.join(os.environ['HOME'], "amir-nav-experiments/maze_files/caffe_room/") 
    
    maze_path = base_path + '/Maze_dataset/'
    result_path = base_path + "/Multy_Maze_results/"
    maze_ind = 0
    goal_number = 10
    # maze_path = None
    goal_order_list = []
    last_temperature = 2.0
    for i in xrange(maze_ind, goal_number):
        experiment = FAKE_test(base_path, maze_path, maze_ind, result_path, i)
        print "last temperature", last_temperature
        experiment.RL.temperature = last_temperature
        try:
            while True:
                experiment.runner_online()
            
        except KeyboardInterrupt:
            last_temperature = experiment.RL.temperature
            experiment.save_state(force = True)
            goal_order_list.append(experiment.active_goal_list)
            print "Goal %d done" % i
            try:
            	os.remove(base_path+'nfq/analysis.txt')
            except: 
                pass

            os.remove(base_path+'nfq/network_.net')
            os.remove(base_path+'nfq/progress')
            os.remove(base_path+'nfq/total_reward')
            os.remove(base_path+'nfq/weights.txt')
          
        experiment.reset()
        del experiment
    a = np.asarray(goal_order_list)
    np.save(maze_path + '/singlegoal_goal_order.npy', a)
    rospy.signal_shutdown("All goals are reached")
    
