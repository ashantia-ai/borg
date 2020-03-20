#!/usr/bin/env python
import matplotlib.pyplot as plt
import time
import cPickle, sys
import math, numpy, cv2

import os
import Queue

class NN():
    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        
        self.w = numpy.asarray(numpy.random.RandomState().uniform(
                    low=100,
                    high=100,
                    size=(self.n_input, self.n_output)), dtype=numpy.float64)
        self.b = numpy.zeros((self.n_output,1))
        
        self.learning_rate = 0.2
        
        self.batch_update = []
        self.batch_data = []
        self.batch_labels = []
        self.replay_data = []
        
    
    def out(self, input):
        return numpy.dot(input, self.w)
    
    def mse(self, input, target):
        return numpy.mean(target - (self.out(input)) ** 2)
    
    def online_update(self, input, target):
        #input = numpy.reshape(input.size1)
        delta_ey = (target - self.out(input))
        new_w = self.w + self.learning_rate * numpy.dot(input.T, delta_ey)
        self.w = new_w
        
    def batch(self, input, target):
        delta_ey = (target - self.out(input))
        self.batch_update.append(self.learning_rate * numpy.dot(input.T, delta_ey))
        self.batch_data.append(input)
        self.batch_labels.append(target)
        
    def update(self):
        
        data = zip(self.batch_data, self.batch_labels)
        for i in range(50):
            update = [self.learning_rate * numpy.dot(input.T, target - self.out(input)) for input, target in data]
            updates = numpy.asarray(self.batch_update)
            self.w += numpy.mean(updates, axis = 0)
        self.batch_update = []
        self.batch_data = []
        self.batch_labels = []
        
        
         
class FAKE_test(object):
    def __init__(self):
        
        #Prepares Action-Service for NFQ 
        
        self.discount_factor = 0.9
        self.success_trials = 0
        self.failed_trials = 0
        
        self.step_number = 0
        self.total_number = 0
        self.batch_update = 64
        self.trial_max_steps = 1000
        
        self.__prepare_init_n_goal_position()
        self.__prepare_rewards()
        self.__prepare_action_variables()
        self.__prepare_RL_variables()
        
        self.network = NN(n_input = 54, n_output = 4)        
        self.reset()
        
        plt.figure(1)
        plt.axis([0, 1000000, 0, 1000])
        plt.ion()
        plt.show(block=False)
    
    def __easy_grid(self, x, y):
        '''
        Converts the location output to grid like format
        '''

        rows, columns = self.structure.shape
        depth = 1 #2 * Pi / 1.57
        
        x_idx = x
        y_idx = y
        
        #print "x: %s, y: %s" % (x_idx,y_idx)
        pos_matrix = numpy.zeros((rows, columns), dtype=numpy.float64)
        pos_matrix[self.x, self.y] = 1.0
        
        return numpy.reshape(pos_matrix, (1, pos_matrix.size))
        
    def __perform_simple_action(self, action_num):
        x = self.x
        y = self.y
        
        if self.structure[self.x, self.y] == 1:
            print "INSIDE OBJECTS."

        
        #print "Current Location is X: %s, Y: %s. Action is : %s " % (x, y, action_num)
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
            return False
        
        self.x = x
        self.y = y
        
        
        return True
        
    def __prepare_action_variables(self):
        #TODO: needs to called from set_params too
        #Action Variables
        self.in_collision = False
        
               
    def __prepare_init_n_goal_position(self):
        '''
        self.structure =  numpy.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                       [1, 0, 0, 1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 1, 0, 0, 1, 0, 1],
                                       [1, 0, 0, 1, 0, 0, 1, 0, 1],
                                       [1, 0, 0, 1, 0, 1, 1, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 1, 0, 1],
                                       [1, 1, 1, 1, 1, 1, 1, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        '''
        self.structure =  numpy.array([[0, 0, 0, 0, 0, 0, 0, 1, 0],
                                       [0, 0, 1, 0, 0, 0, 0, 1, 0],
                                       [0, 0, 1, 0, 0, 0, 0, 1, 0],
                                       [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        '''
        for x in range(self.structure.shape[0]):
            print "\n"
            for y in range(self.structure.shape[1]):
                print self.structure[x,y],", ",
        '''
        
        self.x = 2
        self.y = 0
        
        #self.goal = {'x':3.66, 'y':-0.59, 'theta':-90} #Next to stove, facing to it
        self.goal = {'x':0, 'y':8} # infront of init position.
        
        self.goal_xy_threshold = 0 #Meters
        
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
        
        
    def __prepare_RL_variables(self):
        nfq_action_outputs = 4
        
        self.trial_length = 300 #seconds
        self.trial_begin = time.time()
        self.trial_time = lambda : time.time() - self.trial_begin 
        self.running = True
                    
    def reset(self):
        
        
        self.__prepare_init_n_goal_position()
        self.in_collision = False
        self.last_action = None
        self.last_reward = None
        self.last_state = None
        print self.step_number
        self.step_number = 0
        self.trial_begin = time.time()
        self.running = True
        
    def runner(self):
        if self.running:
            self.step_number += 1
            self.total_number += 1
            
            state = self.get_state_representation()
            
            q_values = self.network.out(state)
            action = numpy.argmax(q_values)
            if numpy.random.random() < 0.1:
                action = numpy.random.randint(0,4)                
            self.__perform_simple_action(action)
            
            new_state = self.get_state_representation()
            new_q_values = self.network.out(new_state)        
            reward = self.get_reward()

            times_up = self.step_number >= self.trial_max_steps
            
            if reward >= self.angular_fix_reward: 
                self.success_trials += 1
                new_q = reward
                q_values[0, action] = new_q
                self.network.online_update(state, q_values)

            elif times_up:
                self.failed_trials += 1
                reward = self.trial_fail_punishment #Punishment for not reaching goal
                new_q = reward + self.discount_factor * numpy.max(new_q_values)
                q_values[0, action] = new_q
                self.network.online_update(state, q_values)
            else:
                new_q = reward + self.discount_factor * numpy.max(new_q_values)
                q_values[0, action] = new_q
                self.network.online_update(state, q_values)
            
            if self.total_number % 10000 == 0:
                plt.draw()    
                    
            if times_up or reward >= self.angular_fix_reward:
                if (self.failed_trials == 0 and self.success_trials == 0):
                    pass #This happens after a batch update
                else:
                    print "SUCCESS RATIO: ", float(self.success_trials)/float(self.success_trials + self.failed_trials)
                self.running = False
                
                plt.figure(1)
                plt.scatter(self.total_number, self.step_number)
                
                numpy.savetxt("/home/amir/nfq/manual.txt", self.network.w,fmt='%8.2f', delimiter=',', header="\n---BEGIN---\n", footer="\n----END----\n")
            
                self.reset()    
    
    def runner_batch(self):
        if self.running:
            self.step_number += 1
            self.total_number += 1

            
            state = self.get_state_representation()
            
            q_values = self.network.out(state)
            action = numpy.argmax(q_values)
            if numpy.random.random() < 0.1:
                action = numpy.random.randint(0,4)                
            self.__perform_simple_action(action)
            
            new_state = self.get_state_representation()
            new_q_values = self.network.out(new_state)        
            reward = self.get_reward()

            times_up = self.step_number >= self.trial_max_steps
            
            if reward >= self.angular_fix_reward: 
                self.success_trials += 1
                new_q = reward
                q_values[0, action] = new_q
                self.network.batch(state, q_values)

            elif times_up:
                self.failed_trials += 1
                reward = self.trial_fail_punishment #Punishment for not reaching goal
                new_q = reward + self.discount_factor * numpy.max(new_q_values)
                q_values[0, action] = new_q
                self.network.batch(state, q_values)
            else:
                new_q = reward + self.discount_factor * numpy.max(new_q_values)
                q_values[0, action] = new_q
                self.network.batch(state, q_values)
                
                
            
            if self.total_number % self.batch_update == 0:
                self.network.update()
                
            if self.total_number % 10000 == 0:
                plt.draw()
                    
            if times_up or reward >= self.angular_fix_reward:
                if (self.failed_trials == 0 and self.success_trials == 0):
                    pass #This happens after a batch update
                else:
                    print "SUCCESS RATIO: ", float(self.success_trials)/float(self.success_trials + self.failed_trials)
                self.running = False
                
                numpy.savetxt("/home/amir/nfq/manual.txt", self.network.w,fmt='%8.2f', delimiter=',', header="\n---BEGIN---\n", footer="\n----END----\n")
                
                plt.figure(1)
                plt.scatter(self.total_number, self.step_number)

            
                self.reset()    

    def get_state_representation(self):
        res = self.__easy_grid(self.x, self.y)    
        return res
        
    def get_reward(self):
        reward = 0
        
        if self.in_collision:
            self.in_collision = False
            return self.obstacle_punishment
        
        d_location = (abs(self.goal['x'] - self.x) + abs(self.goal['y'] - self.y) ) 
                 
        #Check location distance for reward
        if d_location <= self.goal_xy_threshold:
            reward = self.angular_fix_reward
            print "total actions:", self.total_number
        else:
            return reward + self.fix_punishment

        return reward
    
        
if __name__ == '__main__': 
    
    experiment = FAKE_test()
    while True:
        experiment.runner_batch()