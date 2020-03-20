import theano
from theano import tensor as T, config, shared
import rl_methods.linear_regression as linear_regression
from random import shuffle
from copy import deepcopy

import rospy
import numpy
import os
import cPickle
import pickle
import dill
import time
import math

from nfq import NFQ

################################################################################
# # CONSTANTS                                                                  ##
################################################################################
EXPLORE_GREEDY = 1  # : E-Greedy Exploration: probability of random action
EXPLORE_GAUSSIAN = 2  # : Gaussian exploration around current estimate
EXPLORE_BOLZMANN = 3
EXPLORE_NAVIGATION = 4

p_ups = 0
n_ups = 0


class MGNFQ(NFQ):

    def __init__(self, base_path, num_outputs, num_inputs = 54, maze_shape = (10, 10), **kwargs):
        
        super(MGNFQ, self).__init__(base_path, num_outputs, num_inputs = 54, **kwargs)
        
        self.network_filename = "network.net"
        
        self.rl_method = "qlearning"
            
        ########################################################################
        # # RL PARAMETERS                                             ##
        ########################################################################
        self.epsilon = 0.1  # Epsilon for e-greedy exploration
        self.min_epsilon = 0.1  # Minimum exploration
        self.td_epsilon = 0.0  # The variable value to explore base on neighboring td error
        
        self.discount_factor = 0.9  # Decay of the reward
        
        self.shared_data = []
        self.shared_labels = [] 
        
        # Q-map and TD-Error Map of size map_x,map_y, action_num (self.num_outputs)
        self.map_x, self.map_y = maze_shape
        self.map_x = int(self.map_x)
        self.map_y = int(self.map_y)
        self.q_map = []
        self.td_map = []
        self.goals_td_map = []
        
        self.max_replay_size = 100000
        
        if kwargs:
            self.set_parameters(**kwargs)
        
        self.network_number = 10
        self.network_folder = "mg_nets"
            
    def calculate_updates(self, list = None):
        global p_ups
        p_ups = 0
        global n_ups
        n_ups = 0

        def values(history):
            global p_ups
            global n_ups
            lstate, laction, lq_values, reward, q_values, action = history            
            
            if numpy.max(lstate) != 1:
                raise 

            if action == -10:
                if reward < 0:
                    # print update
                    n_ups += 1
                else:
                    p_ups += 1
                lq_values[laction] = reward 
            else:
                update = reward + self.discount_factor * numpy.max(q_values)
                
                if reward < 0:
                    # print update
                    n_ups += 1
                else:
                    p_ups += 1
                    # print update
                
                lq_values[laction] = update
                
            return lstate, lq_values
            
        data = []
        label = []
        
        if list != None:
            state_history = list
        else:
            state_history = self.state_history
        
        data, label = zip(*map(values, state_history))
        
        data = numpy.asarray(data, dtype = theano.config.floatX)
        label = numpy.asarray(label, dtype = theano.config.floatX)
        
        print data.shape
        # lf = open("/home/amir/nfq/label.txt", 'a')
        # numpy.savetxt(lf, label, fmt='%8.3f', delimiter=',', header="\n---BEGIN---\n", footer="\n----END----\n")
        # lf.close()
        
        data = theano.shared(data)
        label = theano.shared(label)

        return data, label
            
    def experience_update(self, tuple):
        state, action, reward, next_state = tuple      
        
        old_values = self.network_output(numpy.reshape(state, (1, state.size)))[0]

        if next_state == None:
            update = reward
        else:
            values = self.network_output(numpy.reshape(next_state, (1, next_state.size)))[0]
            update = reward + self.discount_factor * numpy.max(values)
            
        old_values[action] = update
        
        return state, old_values
            
    def exploration(self, estimate):
        self.frames_progress += 1
        #print "qvalues:" , estimate
        if self.explore_strategy == EXPLORE_GREEDY:
            if self.test_performance:
                explore = numpy.random.random(1)[0] <= self.min_epsilon
            else:
                if self.frames_progress <= self.frame_progress_ceiling:
                    explore = True
                else:
                    explore = numpy.random.random(1)[0] <= self.epsilon
            if explore:
                ran_values = numpy.random.random(len(estimate))
                result = numpy.argmax(ran_values)
            else:
                result = numpy.argmax(estimate)
                
        if self.explore_strategy == EXPLORE_BOLZMANN:
            estimate = numpy.asarray(estimate, dtype = numpy.float128)
            
            exp_estimate = numpy.exp(estimate / self.temperature) / numpy.sum(numpy.exp(estimate / self.temperature))
            cum_prob = 0
            rand_value = numpy.random.random()
            
            if rand_value < self.random_action_replacement:
                ran_values = numpy.random.random(len(estimate))
                result = numpy.argmax(ran_values)
            else:
                rand_value = numpy.random.random()
                for idx, value in enumerate(exp_estimate):
                    cum_prob += value
                    if rand_value <= cum_prob:
                        #print "selected action:", idx
                        return idx
                print "temperature", self.temperature
                print "bolzmann:" , exp_estimate    



                return exp_estimate.size - 1
        
        return result
        
    def replay_experience(self):
        history = self.full_history[-min(self.max_replay_size, len(self.full_history)):-1]
        current_progress = self.progress
        current_actions = self.total_actions

        shuffle(history)
        self.update_networks(len(history), history, self.learning_rate / 10)
            
        print "----------------Replay Finished--------------"
        
        del history
        self.progress = current_progress
        self.total_actions = current_actions
        self.state_history = []
        self.full_history = self.full_history[-min(self.max_replay_size, len(self.full_history)):-1]
        
    def load_networks(self, foldername):
        import re
        base = os.path.join(self.base_path, 'network')
        if not os.path.exists(base):
            try:
                os.mkdir(base)
            except Exception as e:
                print repr(e)
        networks_weights = []
        self.td_map = []
        file_counter = 0
        # TODO: Networks are perhaps not being loaded in a correct manner. Assign each network to the current value
        
        try:
            net_num = len(os.listdir(base))
        except: 
            os.makedirs(base)
            net_num = len(os.listdir(base))
        networks_weights = [None] * net_num
        self.td_map = [None] * net_num
        for root, dirs, names in os.walk(base):
            m = re.search("\d", root)
            if m:
                index = int(root[m.start():])
            
            for filename in names:
                # Actually checking for name rather than exception.
                # TODO:change the names to avoid confusion
                ext_start = filename.rfind('.')
                ext = filename[:ext_start]
                if ext == 'network':
                    filepath = os.path.join(root, filename)
                    try:
                        f = open(filepath, 'rb')
                        instance = cPickle.load(f)
                        f.close()
                        networks_weights[index] = instance
                    except Exception as e:
                        print repr(e)
                        raise e
                if ext == 'td_map':
                    filepath = os.path.join(root, filename)
                    try:
                        data = numpy.load(filepath)
                        self.td_map[index] = data
                    except Exception as e:
                        print repr(e)
                        raise e
        
        return networks_weights
    
    def save_state(self):
        base_path = os.path.join(self.base_path)
        filepath = os.path.join(base_path, 'mgnfq_state')
        print "Saving MGNFQ States to File..."
        try:
        
            f = file(filepath, 'wb')
            dill.dump(self.__dict__, f)
            
            print "Saving Succeeded"
        except Exception as ex:
            print ex
        
        
        #pickle.dump(self.__dict__, f, protocol = 0)
        
    def load_state(self, reset_xp_arrays = False):
        base_path = os.path.join(self.base_path)
        filepath = os.path.join(base_path, 'mgnfq_state')
        
        try:
            f = file(filepath, 'rb')
            self.__dict__.update(dill.load(f))
        except Exception as ex:
            print "Loading of states went wrong"
            print ex
            
        if reset_xp_arrays:
            self.rl_data = []
            self.rl_labels = []
            self.rl_replay_data = []
            
    def save_network(self, foldername):
        # TODO:Also save TD_map
        base_path = os.path.join(self.base_path, 'network')
        
        # filepath = os.path.join(base_path, curfile)
        for i in range(len(self.network)):
            dirname = 'net%d' % i
            filepath = os.path.join(base_path, dirname, self.network_filename)
            td_filename = "td_map"
            td_filepath = os.path.join(base_path, dirname, td_filename)
            q_filename = "q_map"
            q_filepath = os.path.join(base_path, dirname, q_filename)
            if not os.path.exists(os.path.dirname(filepath)):
                try:
                    os.makedirs(os.path.dirname(filepath))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
        
            f = file(filepath, 'wb')
            numpy.save(td_filepath, self.td_map[i])
            # numpy.save(q_filepath, self.q_map[i])
            list = []
            for param in self.network[i].params:
                # print param.get_value(borrow=True)
                list.append(param.get_value(borrow = True))
                
            cPickle.dump(list, f, protocol = cPickle.HIGHEST_PROTOCOL)
            f.close()

    def select_action(self, state, net_number = 0):
        if not hasattr(self, 'network'):
            self.setup_ann(state)
        q_values = self.network_output[net_number](numpy.reshape(state, (1, state.size)))[0]
        return self.exploration(q_values)
       
    def setup_ann(self, nn_input):
        """
        This function loads existing networks or creates new networks for the
        actor and the critic.
        """
        try:
            num_inputs = int(nn_input)
        except:
            num_inputs = len(nn_input)
            
        # Lists for networks, their output, and training functions.
        self.network = []
        self.network_output = []
        self.train_network = []
        self.train_batch = []
        self.x = []
        self.y = []
        self.ox = []
        self.oy = []
        index = []
        #----
            
        activation_func = self.activation_function
        rng = numpy.random.RandomState()
        
        # Neural network initial weights and biases
        fixed_init_weight = 80  # This initial weight encourages exploration since Q-values will be the same and high
        W_init = numpy.asarray(rng.uniform(
                            low = fixed_init_weight,
                            high = fixed_init_weight,
                            size = (num_inputs, self.num_outputs)), dtype = theano.config.floatX)  # @UndefinedVariable
                                
        b_init = numpy.zeros((self.num_outputs,), dtype = theano.config.floatX)
        
        # TODO: network weights are a list, it is being added incorrectly my friend. FIX IT
        network_weights = self.load_networks(self.network_folder)
        # TODO: Warning I think the symbolic inputs should also be separated
        for i in range(self.network_number):
            
            self.shared_data.append(theano.shared(value = numpy.zeros((1000, num_inputs), dtype = T.config.floatX)))
            self.shared_labels.append(theano.shared(value = numpy.zeros((1000, self.num_outputs), dtype = T.config.floatX)))
            # i is the index for the networks and their output, and train functions
            
            self.x.append(T.matrix())  # symbolic input list
                
            if network_weights == None or len(network_weights) == 0:
                rospy.logwarn("No networks to load. Initializing all the weights.")
                current_weight = [W_init, b_init]
                # TODO: Perhaps only keep the td_map, q_map can be made immediately from the network itself, basically the weights of the net
                self.q_map.append(numpy.ones((self.map_x, self.map_y, self.num_outputs), dtype = numpy.float32) * (fixed_init_weight))
                self.td_map.append(numpy.ones((self.map_x, self.map_y, self.num_outputs), dtype = numpy.float32) * +1200)  # +1200 is a high value for TD. It means not traversed.
            else:
                current_weight = network_weights[i]
            self.network.append(linear_regression.NNRegression(input_data = self.x[i], n_in = num_inputs,
                                                              n_out = self.num_outputs, weights = current_weight , act_func = activation_func))
            
            self.network_output.append(theano.function([self.x[i]],
                outputs = self.network[i].output, allow_input_downcast = True))
    
            # cost = self.network.cost(self.y)
            self.ox.append(T.fvector())
            self.oy.append(T.fvector())
    
            cost = self.network[i].online_cost(self.ox[i], self.oy[i])
         
            # compute the gradient of cost with respect to theta (sotred in params)
            # the resulting gradients will be stored in a list gparams
            gparams = []
            for param in self.network[i].params:
                gparam = T.grad(cost, param)
                gparams.append(gparam)

            # specify how to update the parameters of the model as a list of
            # (variable, update expression) pairs
            updates = []
            for param, gparam in zip(self.network[i].params, gparams):
                updates.append((param, param - self.learning_rate * gparam))
        
            # compiling a Theano function `train_model` that returns the cost, but
            # in the same time updates the parameter of the model based on the rules
            # defined in `updates`

            self.train_network.append(theano.function(inputs = [self.ox[i], self.oy[i]], outputs = cost,
                    updates = updates, allow_input_downcast = True))
        
            ###BATCH NETWORK########
            self.y.append(T.fmatrix())
            index.append(T.lscalar())
            
            cost = self.network[i].cost(self.y[i])
             
            # compute the gradient of cost with respect to theta (sotred in params)
            # the resulting gradients will be stored in a list gparams
            gparams = []
            for param in self.network[i].params:
                gparam = T.grad(cost, param)
                gparams.append(gparam)    
                # specify how to update the parameters of the model as a list of
                # (variable, update expression) pairs
            updates = []
            for param, gparam in zip(self.network[i].params, gparams):
                updates.append((param, param - self.learning_rate * gparam))
            
            # compiling a Theano function `train_model` that returns the cost, but
            # in the same time updates the parameter of the model based on the rules
            # defined in `updates`
            self.train_batch.append(theano.function(inputs = [index[i]], outputs = cost,
                    updates = updates,
                    givens = {
                        self.x[i]: self.shared_data[i][index[i] * self.batch_size: (index[i] + 1) * self.batch_size],
                        self.y[i]: self.shared_labels[i][index[i] * self.batch_size: (index[i] + 1) * self.batch_size]}))

    def set_parameters(self, **kwargs):
        """
        Set the parameters for the algorithm. Each parameter may be specified as
        a keyword argument. Available parameters are:

        alpha               The learning rate for the Critic in [0, 1]
        beta                The learning rate for the Actor in [0, 1]
        epsilon             The exploration probability for 
                            e-greedy exploration in [0, 1]
        sigma               The standard deviation for Gaussian exploration > 0
        discount_factor     The value attributed to future rewards in (0, 1)
        random_decay        The decay of the epsilon and sigma paramters after
                            each save of the algorithm. This is the factor
                            the value is multiplied with. Should be in [0, 1]
        explore_Strategy    EXPLORE_GAUSSIAN or EXPLORE_GREEDY
        num_outputs         The number of outputs required from the actor. This
                            should be an integer greater than 0.
        ensemble_size       The number of Neural Networks to use in ensemble to
                            optimize the output of the actor and critic. The
                            output of these networks is averaged to obtain the
                            next action.
        td_var              The initial variance of the TD-error. Default 1.0
        var_beta            The factor of the update of the running average of the
                            varianceof the TD-error. Default: 0.001
        """
        for key, value in kwargs.iteritems():
            if key == "activation_function":
                if value == 0:
                    self.activation_function = T.nnet.sigmoid
                elif value == 1:
                    self.activation_function = None
                elif value == 2:
                    self.activation_function = T.nnet.relu
            elif key == "explore_strategy":
                if value == EXPLORE_GAUSSIAN or \
                    value == EXPLORE_GREEDY or \
                    value == EXPLORE_NAVIGATION or value == EXPLORE_BOLZMANN:
                    self.explore_strategy = value
                else:
                    raise Exception("Invalid exploration strategy %d" % value)
            elif key == "num_outputs":
                self.num_outputs = int(value)
            elif key == "sigma" and (type(value) is tuple or type(value) is list):
                self.min_sigma, self.sigma = value
            elif key == "epsilon" and (type(value) is tuple or type(value) is list):
                self.min_epsilon, self.epsilon = value
            elif key == "max_replay_size":
                self.max_replay_size = int(value)
            elif key in self.__dict__:
                self.__dict__[key] = float(value)
            elif not key in self.__dict__:
                raise Exception("Unknown setting: %s = %s" % (key, repr(value)))
  
    def update_networks_plus(self, learning_rate = None, experience_replay = False, train_epochs = 1000, desired_error = 0.00001, reached_goals = []):
        # print "Getting Values"
        
        # self.rl_data =self.rl_data[-min(self.max_replay_size, len(self.rl_data)):-1]
        # self.rl_labels = self.rl_labels[-min(self.max_replay_size, len(self.rl_labels)):-1]
        self.rl_replay_data = self.rl_replay_data[-self.max_replay_size:]
        print "Replay Data Length", len(self.rl_replay_data)
        size = self.batch_size
        
        # TODO: Warning this will be nonesense for multi goal. Fix it
        f = open(self.weights_file, 'a')
        
        average_error = 0
        total_average_error = 0
        last_error = 0
        epochs = 0
        state = aciton = reward = new_state = None
        
        if experience_replay:
            # return #WHYYYYYYYYYYYYYYYYY?
            print "EXPERIENCE REPLAY INITIATED"
            rl_replay_data_separate = []
            for i in range(self.network_number):
                matches = [ data for data in self.rl_replay_data if data[-1] == i]
                if matches:
                    rl_replay_data_separate.append(matches)
                
            self.rl_replay_data = self.rl_replay_data[-self.max_replay_size:]
            # TODO: For each network correctly shuffle data separately and initiate experience replay
            
            for replay_data in rl_replay_data_separate:
                shuffle(replay_data)    
                state, action, reward, new_state, network_number = zip(*replay_data)
            
                network_number = network_number[0]
                if reached_goals and reached_goals[network_number] == 1:
                    #print "skipping learned goal No. ", network_number
                    pass
                else:
                    print "Reached Goals List: ", reached_goals
                    print "Network Number: ", network_number
                state = numpy.asarray(state, dtype = T.config.floatX)
            
                if type(new_state) is tuple:
                    new_state = numpy.asarray(list(new_state), dtype = T.config.floatX)
                else:
                    try:
                        new_state = numpy.asarray(new_state, dtype = T.config.floatX)
                    except Exception as e:
                            raise e
                reward = numpy.asarray(reward, dtype = T.config.floatX)
                
                reached_indices = numpy.where(reward == 100)     
                # while True:
                average_error = 0
                epochs += 1
                # Extract new q-values based on old experiences
                current_q_values = self.network_output[network_number](state)
                next_q_values = self.network_output[network_number](new_state)
                
                # TODO: WARNING, we are considering the max reward as 100. Do not change the reward
                                
                temp_updates = reward + self.discount_factor * numpy.max(next_q_values, axis = 1)
                temp_updates[reached_indices] = 100.0
                    
                for idx, ac in enumerate(action):
                    current_q_values[idx, ac] = temp_updates[idx]
                
                labels = numpy.asarray(current_q_values, dtype = theano.config.floatX)
                self.shared_data[network_number].set_value(state)
                self.shared_labels[network_number].set_value(labels)
                
                self.n_train_batches = self.shared_data[network_number].get_value(borrow = True).shape[0] / size
                
                for minibatch_index in xrange(self.n_train_batches):
                    minibatch_avg_cost = self.train_batch[network_number](minibatch_index)
                    average_error += minibatch_avg_cost
                    
                if self.n_train_batches != 0:
                    average_error /= (self.n_train_batches)
                
                if math.fabs(average_error) < desired_error or epochs >= train_epochs:
                   print "Average Batch Error: %3.10f -- %3.10f" % (average_error, last_error)
                   # break
                else:
                    last_error = average_error
                    # print "Average Batch Error:", average_error
                
                print "Average Batch Error: %3.6f" % (average_error)
                total_average_error += average_error
                self.shared_data[network_number].set_value([[]])
                self.shared_labels[network_number].set_value([[]])
        else:  
            
            rl_data_separate = []
            for i in range(self.network_number):
                matches = [ data for data in self.rl_data if data[-1] == i]
                if matches:
                    rl_data_separate.append(matches)
            rl_labels_separate = []
            for i in range(self.network_number):
                matches = [ data for data in self.rl_labels if data[-1] == i]
                if matches:
                    rl_labels_separate.append(matches)
            temp_data = None
            temp_labels = None 
            for rl_data, rl_labels in zip(rl_data_separate, rl_labels_separate):
                # TODO: Take into account the empty network goals
                rl_data, network_number = zip(*rl_data)
                rl_labels, _ = zip(*rl_labels)
                
                # Network number list is all similar. Taking the first one
                network_number = network_number[0]
                epochs = 0
                
                data = numpy.asarray(rl_data, dtype = theano.config.floatX)
                labels = numpy.asarray(rl_labels, dtype = theano.config.floatX)
                self.shared_data[network_number].set_value(data, borrow = True)    
                self.shared_labels[network_number].set_value(labels, borrow = True)
                
                while True:
                    
                    # TODO: For each network, perform batch training and save analysis to correct folders
                    average_error = 0
                    epochs += 1
                    
                    self.n_train_batches = self.shared_data[network_number].get_value(borrow = True).shape[0] / size
                    
                    for minibatch_index in xrange(self.n_train_batches):
                        minibatch_avg_cost = self.train_batch[network_number](minibatch_index)
                        print "minibatch err: ", minibatch_avg_cost
                        average_error += minibatch_avg_cost
                        
                    if self.n_train_batches != 0:
                        average_error /= (self.n_train_batches)
                    
                    if math.fabs(average_error) < desired_error or epochs >= train_epochs:
                       # print "Average Batch Error: %3.6f -- %3.6f" % (average_error, last_error)
                       break
                    else:
                        last_error = average_error
        self.rl_data = []
        self.rl_labels = []       
        return total_average_error
        
        
    def update_online(self, state_list, action, reward_list, next_state_list, active_goal = 0, reached_goals=[]):
        # state, action, reward, next_state
        # (self, state_list, action, reward_list, next_state_list, active_goal = 0)
        if not hasattr(self, 'network'):
            self.setup_ann(state_list[active_goal])
        
        # State, reward, next state list
        s_r_ns_list = zip(state_list, reward_list, next_state_list, range(self.network_number))
        
        cg_processed = False
        for state, reward, next_state, network_number in s_r_ns_list:
            
            if len(reached_goals) > 0 and reached_goals[network_number] == 1:
                pass   
            #reconstruct flattened state in to a matrix
            reconst_state = numpy.reshape(state, (self.map_x, self.map_y))
            # TODO: Only works with 2D matrices, think about multi dimensional
            #find x,y for the current state that we are at.
            x, y = numpy.where(reconst_state == 1)
            
            # Q-values of current_state which we call old values
            old_values = self.network_output[network_number](numpy.reshape(state, (1, state.size)))[0]
            
            #If we are in the goal state, the value of the state should be full reward.
            if reward >= 100:
                update = 100
                td_err = (update - old_values[action]) ** 2
                self.td_map[network_number][x, y, action] = td_err
            else:
                # Q-values of next state
                values = self.network_output[network_number](numpy.reshape(next_state, (1, next_state.size)))[0]
                update = reward + self.discount_factor * numpy.max(values)                
                td_err = (update - old_values[action]) ** 2
                self.td_map[network_number][x, y, action] = td_err
            
            old_values[action] = update
            self.rl_replay_data.append((state, action, reward, next_state, network_number))
        
            state = numpy.asarray(state, T.config.floatX)
            new_values = numpy.asarray(old_values, dtype = T.config.floatX)
        
            self.train_network[network_number](state, new_values)
        
            if not cg_processed and network_number == active_goal:
                self.total_reward += (self.discount_factor ** self.steps) * reward
                self.cum_reward += reward
                self.total_actions += 1
                cg_processed = True

        self.steps += 1
            
    def update_batch(self, state_list, action, reward_list, next_state_list, active_goal = 0):
        if not hasattr(self, 'network'):
            self.setup_ann(state_list[active_goal])
        
        # State, reward, next state list
        s_r_ns_list = zip(state_list, reward_list, next_state_list, range(self.network_number))
        
        # First tuple is the current goal
        cg_processed = False
        counter = 0
        for state, reward, next_state, network_number in s_r_ns_list:
            reconst_state = numpy.reshape(state, (self.map_x, self.map_y))
            # TODO: Only works with 2D matrices, think about multi dimensional
            x, y = numpy.where(reconst_state == 1)
            # Q-values of current_state which we call old values
            old_values = self.network_output[network_number](numpy.reshape(state, (1, state.size)))[0]
            if reward == 100:
                update = reward
                td_err = (update - old_values[action]) ** 2

                self.td_map[network_number][x, y, action] = td_err
            else:
                try:
                    # Q-values of next state
                    values = self.network_output[network_number](numpy.reshape(next_state, (1, next_state.size)))[0]
                except Exception as e:
                    print "State size", state.size
                    print "next_state" , next_state
                    print "counter", counter
                    print "the full list:\n", s_r_ns_list
                    print "next state size" , next_state.size
                    raise e
                # The Q-learning update formula
                update = reward + self.discount_factor * numpy.max(values)                
                td_err = (update - old_values[action]) ** 2
                self.td_map[network_number][x, y, action] = td_err
                
            counter += 1
            old_values[action] = update
        
            # The rl_data and rl_labels were meant for single goal batch update. I add a network_number to distinguish multiple goals
            self.rl_data.append((state, network_number))
            self.rl_labels.append((old_values, network_number))
            
            self.rl_replay_data.append((state, action, reward, next_state, network_number))
            
            # the reward for the current goal is only calculated as part of total_reward / cum_reward
            if not cg_processed and network_number == active_goal:
                self.total_reward += (self.discount_factor ** self.steps) * reward
                self.cum_reward += reward
                self.total_actions += 1
                cg_processed = True

        self.steps += 1

    def reset(self):
        """
        This method will reset the Neural Fitten Q Iteration, meaning it will be in the
        first state of a sequence. During the first round, no training will
        occur and only an action will be generated.
        """
        self.last_output = None
        self.last_value = None
        self.last_state = None
        if self.test_performance:
            rew_file = os.path.basename(self.reward_file)
            rew_dir = os.path.dirname(self.reward_file)
            rew_file = os.path.join(rew_dir, "test_" + rew_file)
            open(rew_file, 'a').write("%d %f\n" % (self.test_count, self.total_reward))
        else:
            open(self.reward_file, 'a').write("%d %f\n" % (self.progress, self.total_reward))
        self.total_reward_list.append(self.total_reward)
        self.total_reward = 0
        # self.cum_reward = 0
        self.steps = 0

    def clear_memory(self):
        self.rl_data = []
        self.rl_labels = []
        self.rl_replay_data = []
        
