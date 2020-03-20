#!/usr/bin/env python
import matplotlib.pyplot as plt
import rospy, roslib
import time, math
import cPickle, sys, copy
import math, numpy, cv2, random
import sd_autoencoder

sys.modules['sd_autoencoder'] = sd_autoencoder
import rl_methods.nfq as nfq
import rl_methods

import theano
from theano import tensor as T, config, shared
from theano.tensor.shared_randomstreams import RandomStreams

#theano.config.profile = True
#theano.config.profile_memory = True

import os
import Queue
import numpy as np
    
class Position(object):
    def __init__(self, x = 0, y = 0, z = 0, roll = 0, pitch = 0, yaw = 0):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw


class Test(object):
    def __init__(self, base_path, Id, **kwargs):
        
        self.show=True
        self.base_path=base_path 
        self.__init_ros_node(Id)
        self.current_maze=0
        self.fig1_data=[]       
        self.fig2_data=[]

        self.reset_all()

        if self.show:
            plt.figure(1)
            #plt.axis([0, 400000, -2, 10])
            plt.ion()
            plt.show(block=False)
            
            plt.figure(2)
            #plt.axis([0, 20000, 0, 1000])
            plt.ion()
            plt.show(block=False)

    def reset_all(self):

        self.steps_number = 0
        self.trial_max_steps = 1000
        
        self.maze_to_show=[]        

        self.__prepare_file_structure(self.base_path)
        self.input_type = "ground_truth" #Options: 1- autoencoder . 2-location, 3- ground truth location, Default: ground truth
        self.method = "nfq" 

        self.__prepare_rewards()
        self.__prepare_action_variables()

        self.__prepare_performance_variables()
        self.__prepare_experience_replay()

        self.__prepare_RL_variables()
       

    def __init_ros_node(self, num=0):
        ''' Initialized the ros node and the main publishers
        '''
        rospy.init_node('Fake_nav_'+str(num))
        rospy.loginfo("Node initialized")


    def __prepare_file_structure(self,base_path):

        self.base_path = base_path
        #self.__create_directories()


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


    def __prepare_action_variables(self):
        #TODO: needs to called from set_params too
        #Action Variables
        self.in_collision = False
        if self.method == "cacla":
            self.previous_action = [0,0]
        else:
            self.previous_action = 0


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


    def __prepare_RL_variables(self):
        action_outputs = 2
        nfq_action_outputs = 4
        
        self.RL = rl_methods.nfq.NFQ(os.path.join(self.base_path, "nfq"), nfq_action_outputs)
        
        self.trial_length = 1 #seconds
        self.trial_begin = time.time()
        #print self.trial_begin
        self.trial_time = lambda : time.time() - self.trial_begin 
        self.running = True


    def reset(self):
    
        self.empty_vector = numpy.zeros((1, self.structure.size))
        self.empty_vector = numpy.reshape(self.empty_vector, -1)
        self.x = self.initx = 0
        self.y = self.inity = 0
        self.radian = self.initradian = 0.0

        self.in_collision = False
        self.RL.reset()
        self.trial_begin = time.time()
        self.running = True


    def get_state_representation(self):
        #res = self.__convert_to_grid(self.x, self.y, self.radian)
        res = self.__easy_grid(self.x, self.y, self.radian)
            
        return res


    def __easy_grid(self, x, y, theta, x_len = 8, y_len = 5):
        '''
        Converts the location output to grid like format
        '''

        rows, columns = self.structure.shape
        depth = 1 #2 * Pi / 1.57
        
        x_idx = x
        y_idx = y
        
        rospy.logdebug( "x: %s, y: %s" % (x_idx,y_idx))
        pos_matrix = numpy.zeros((rows, columns), dtype=theano.config.floatX)
        pos_matrix[self.x, self.y] = 1.0
        
        return numpy.reshape(pos_matrix, -1)


    def __perform_simple_action(self, action_num):
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


    def get_reward(self):
        reward = 0
        
        if self.in_collision:
            self.in_collision = False
            return self.obstacle_punishment
        
        d_location = ((abs(self.goal['x'] - self.x)) + (abs(self.goal['y'] - self.y)) ) 
        d_angle = math.fabs(self.goal['theta'] - self.radian)
        
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
                return reward
        return reward


    def load_maze_and_goal(self, path, num, ind=0):

        self.structure = np.load(path+'maze_'+str(num)+'.npy')
        goal = np.load(path+'goal_'+str(num)+'.npy')

        (x_goal, y_goal)=goal[ind]


        to_show=copy.deepcopy(self.structure)
        
        to_show[x_goal, y_goal]=2 
        self.maze_to_show=np.asarray(to_show)   
    
        #plt.figure(3)
        plt.matshow(to_show, fignum=3)

        

        self.empty_vector = numpy.zeros((1, self.structure.size))
        self.empty_vector = numpy.reshape(self.empty_vector, -1)
        self.x = self.initx = 0
        self.y = self.inity = 0
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


    def save_results(self, path, consecutive_succes, actions_numbers):
    
        path=path+'Maze_'+str(self.current_maze)+'/'

        try:
            os.mkdir(path)
        except:
            print 'Existing folder!'

        fig1=np.asarray(self.fig1_data)
        fig2=np.asarray(self.fig2_data)
        succ=np.asarray(consecutive_succes)
        actions=np.asarray(actions_numbers)

        np.save(path+'fig1', fig1)    
        np.save(path+'fig2', fig2)
        np.save(path+'success', succ)
        np.save(path+'actions', actions)
        np.save(path+'maze_with_goal', self.maze_to_show)
        self.current_maze+=1


    def runner_batch(self, save_path):
        consecutive_succes=list(np.zeros(50))
        actions_numbers=list(np.zeros(200))
        while True:
            if self.running:
                self.steps_number += 1
                terminal = False
                skip_update = False
                state = self.get_state_representation() 
                action = self.RL.select_action(state)
                self.__perform_simple_action(action)
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
                    #plt.figure(1)
                    #plt.scatter(self.RL.total_actions, 
                    #            self.RL.cum_reward / self.train_frequency_step)
                    self.fig1_data.append([self.RL.total_actions, 
                                self.RL.cum_reward / self.train_frequency_step])

                    self.RL.cum_reward= 0
                    plt.draw()
                        
                    self.train_frequency_action += self.train_frequency_step 
                    self.RL.update_networks_plus()
                    print self.steps_number
                #if self.RL.total_actions > 10000:
                #    exit()
                
                rospy.logdebug(" Reward: %3.2f     Action: %s", reward, action)
                if times_up or reward >= self.angular_fix_reward:
                    print "Time: ", self.trial_time()
                    print "Actions No: ", self.steps_number
                    
                    actions_numbers.append(self.steps_number)
                    if len(actions_numbers)>200:
                        actions_numbers=actions_numbers[1:]
                    
                    success=float(self.success_trials)/float(self.success_trials + self.failed_trials)
                    
                    consecutive_succes.append(success)
                    if len(consecutive_succes)>50:
                        consecutive_succes=consecutive_succes[1:]    

                    if (self.failed_trials == 0 and self.success_trials == 0):
                        pass #This happens after a batch update
                    else:
                        print "SUCCESS RATIO: ", success
                        pass
                    self.running = False
                    self.RL.save()
                    
                    #plt.figure(2)
                    #plt.scatter(self.RL.progress, self.RL.steps)
                    self.fig2_data.append([self.RL.progress, self.RL.steps])
                    
                    try:
                        self.total_reward_queue.put_nowait(self.RL.total_reward)
                        print 'Working'
                    except Queue.Full:
                        self.analyze_experiment()
                        print 'Nope'
                                        
                    

                    if self.RL.total_actions > self.replay_frequency_action:
                        self.replay_frequency_action += self.replay_frequency_step
                        print "-----------Replay XP------------------"
                        self.RL.update_networks_plus(experience_replay = True)
                        
                    
                        
                    if np.asarray(consecutive_succes).mean() >= 0.999 and self.steps_number < 100:
                        print 'Done with this maze'
                        print "-----------Finish_Maze------------------"
                        self.save_results(save_path, consecutive_succes, actions_numbers)
                        #time.sleep(1)
                        self.fig1_data=[]       
                        self.fig2_data=[]
                        consecutive_succes=list(np.zeros(50))
                        actions_numbers=list(np.zeros(200))
                        self.steps_number = 0    
                        self.reset()
                        break  
                    self.steps_number = 0    
                    self.reset()


if __name__=='__main__':
    base_path = "/home/ashantia/University/experiments/random_maze/"
    dataset_path="/home/ashantia/University/experiments/random_maze/Maze_dataset/"
    results_path="/home/ashantia/University/experiments/random_maze/Maze_results/"
    
    experiment = Test(base_path, "0")
    maze_number = 10
    for curr_data in xrange(1):
        for i in xrange(maze_number):
            experiment.load_maze_and_goal(dataset_path, curr_data, i)
            experiment.runner_batch(results_path)
            experiment.reset()
            experiment.reset_all()    



    


