import theano
from theano import tensor as T, config, shared
import rl_methods.linear_regression as linear_regression
from random import shuffle
from copy import deepcopy

import rospy
import numpy
import os
import cPickle
import time
import math
import dill

################################################################################
## CONSTANTS                                                                  ##
################################################################################
EXPLORE_GREEDY = 1     #: E-Greedy Exploration: probability of random action
EXPLORE_GAUSSIAN = 2   #: Gaussian exploration around current estimate
EXPLORE_BOLZMANN = 3
EXPLORE_NAVIGATION = 4

p_ups = 0
n_ups = 0

class NFQ(object):
    def __init__(self, base_path, num_outputs, num_inputs = 54, **kwargs):
        self.base_path = base_path
        self.network_file = os.path.join(base_path, "network.net")
        self.progress_file = os.path.join(base_path, "progress")
        self.reward_file = os.path.join(base_path, "total_reward")
        self.weights_file = os.path.join(base_path, "weights.txt")
        
        self.rl_method = "qlearning"
        self.batch_size = 64 #NN Training is done after N actions is performed
        ########################################################################
        ## NNET PARAMETERS                                                    ##
        ########################################################################
        self.random_range = 0.1
         
        self.activation_function = None #None means linear
        self.num_outputs = num_outputs
        self.learning_rate = 0.3

        ########################################################################
        ## EXPLORATION PARAMETERS                                             ##
        ########################################################################
        self.random_decay = 0.99       # Decay of the epsilon and sigma factors
        self.explore_strategy = EXPLORE_BOLZMANN # Exploration strategy
        self.random_action_replacement = 0.0
        self.temperature = self.init_temperature = 1.0
        
        self.frames_progress = 0
        self.frame_progress_ceiling = 0 #Until what frame choose random actions?
        self.progress = 0               # Number of iterations already  
                                        # performed, useful when restarting.
        try:
            self.progress = int(open(self.progress_file, "r").read())
            print "[%s] Continuing from step %d" \
                  % (self.progress_file, self.progress)
        except:
            self.progress = 0
            
        ########################################################################
        ## RL PARAMETERS                                             ##
        ########################################################################
        self.epsilon = 0.1              # Epsilon for e-greedy exploration
        self.min_epsilon = 0.1          # Minimum exploration
        self.min_temperature = 0.2      # Minimum boltzman temperature
        
        self.discount_factor = 0.9     # Decay of the reward

        ########################################################################
        ## STORAGE FOR USE BY ALGORITHM                                       ##
        ########################################################################
        self.test_performance = False
        
        self.last_output = None
        self.last_value = None
        self.last_state = None
        self.cum_reward = 0
        self.average_reward = lambda : self.cum_reward / self.steps
        self.total_reward = 0
        self.total_reward_list = []
        self.steps = 0
        self.total_actions = 0
        
        ########################################################################
        ## EXPERIENCE REPLAY AND BATCH VARIABLES                              ##
        ########################################################################
        
        self.rl_data = []
        self.rl_labels = []
        self.rl_replay_data = []
        self.rl_replay_labels = []
        
        
        self.state_history = []
        self.full_history = []
        self.success_history = []
        self.max_replay_size = 100000

        if kwargs:
            self.set_parameters(**kwargs)
            
        self.shared_data = theano.shared(name='d', value = numpy.zeros((1000, num_inputs), dtype = T.config.floatX))
        self.shared_labels = theano.shared(name='', value = numpy.zeros((1000, self.num_outputs), dtype = T.config.floatX))
            
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
                    #print update
                    n_ups += 1
                else:
                    p_ups += 1
                lq_values[laction] = reward 
            else:
                update = reward + self.discount_factor * numpy.max(q_values)
                
                if reward < 0:
                    #print update
                    n_ups += 1
                else:
                    p_ups += 1
                    #print update
                
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
        #lf = open("/home/amir/nfq/label.txt", 'a')
        #numpy.savetxt(lf, label, fmt='%8.3f', delimiter=',', header="\n---BEGIN---\n", footer="\n----END----\n")
        #lf.close()
        
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
            estimate = numpy.asarray(estimate, dtype=numpy.float128)
            exp_estimate = numpy.exp(estimate/self.temperature) / numpy.sum(numpy.exp(estimate/self.temperature))
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
                        return idx

                print "bolzman:", exp_estimate
                print "cum_prob:" , cum_prob
                print "rand_value:" , rand_value
                print "last_idx", exp_estimate.size
                return exp_estimate.size - 1
                
        
        return result
    def online_update(self):
        pass  
    def replay_experience(self):
        history = self.full_history[-min(1000000, len(self.full_history)):-1]
        current_progress = self.progress
        current_actions = self.total_actions

        shuffle(history)
        self.update_networks(len(history), history, self.learning_rate / 10)
            
        print "----------------Replay Finished--------------"
        
        del history
        self.progress = current_progress
        self.total_actions = current_actions
        self.state_history = []
        self.full_history = self.full_history[-min(1000000, len(self.full_history)):-1]

    def reset_temperature(self):
        self.temperature = self.init_temperature
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
        #self.cum_reward = 0
        self.steps = 0
        
    def run(self, state, reward, terminal=False, skip_update=False):
        #TODO: record all pairs for NFQ batch learning
        # Check if the ANNs have been set up, and if not, do so.
        if not hasattr(self, 'network'):
            self.setup_ann(state)
            
        
        if terminal:
            lqvalues = self.network_output(numpy.reshape(self.last_state, (1, self.last_state.size)))[0]
            self.state_history.append([self.last_state, self.last_output, lqvalues, reward, None, -10])
            stop_action = 5
            result = stop_action
        else:
            self.current_state = state
    
            # Run the ANNs
            
            q_values = self.network_output(numpy.reshape(state, (1, state.size)))[0]
            
            
            result = self.exploration(q_values)
            
            #Collecting Q-Value sets
            if self.last_output != None:
                lqvalues = self.network_output(numpy.reshape(self.last_state, (1, self.last_state.size)))[0]
                self.state_history.append([self.last_state, self.last_output, lqvalues, reward, q_values, result])
                               
            # Store output for evaluation in the next iteration
            self.last_output = result
            
        self.last_state = state
        self.last_reward = reward
        self.total_reward += (self.discount_factor ** self.steps) *  reward
        self.cum_reward += reward
        self.total_actions += 1

        self.steps += 1

        # We've got the next action, return it
        return result, [0,0] # the [0,0] acts as dummy to be compatible with cacla_nav
        #  @staticmethod
    def load_network(self,filename):
        base = os.path.basename(filename)
        ext_start= base.rfind('.')
        ext = base[ext_start:]
        base = base[:ext_start]
        path = os.path.dirname(filename)
        curfile = "%s_%s" % (base, ext)
        filepath = os.path.join(path, curfile)
        
        try:
            f = open(filepath, 'rb')
            instance = cPickle.load(f)
            f.close()
            rospy.loginfo("Network Loaded Successfully")
            return instance
        except Exception as e:
            print repr(e)
            rospy.logwarn("No networks to load. Starting from scratch")
            return None
    
    #  @staticmethod    
    def save_network(self, filename):
        #TODO:incomplete
        base = os.path.basename(filename)
        ext_start= base.rfind('.')
        ext = base[ext_start:]
        base = base[:ext_start]
        path = os.path.dirname(filename)
        curfile = "%s_%s" % (base, ext)
        filepath = os.path.join(path, curfile)
        
        f = file(filepath, 'wb')
        list = []
        for param in self.network.params:
            list.append(param.get_value(borrow=True))
            
        cPickle.dump(list, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        
    def load_state(self, reset_xp_arrays = False):
        base_path = os.path.join(self.base_path)
        filepath = os.path.join(base_path, 'nfq_state')
        
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
        
    def save_state(self):
        base_path = os.path.join(self.base_path)
        filepath = os.path.join(base_path, 'nfq_state')
        print "Saving NFQ States to File..."
        try:
        
            f = file(filepath, 'wb')
            dill.dump(self.__dict__, f)
            
            print "Saving Succeeded"
        except Exception as ex:
            print ex
        
    def save(self):
        if self.test_performance:
            self.test_count += 1
            prog_file = os.path.basename(self.progress_file)
            prog_dir = os.path.dirname(self.progress_file)
            prog_file = os.path.join(prog_dir, "test_" + prog_file)

            f = open(prog_file, "w")
            f.write("%d" % self.test_count)
            f.close()
            return

        self.save_network(self.network_file)
        #print "Saved Network to %s" % self.network_file
         
        self.epsilon = max(self.min_epsilon, self.epsilon * self.random_decay)
        self.temperature = max(self.min_temperature, self.temperature * self.random_decay)
        
        self.progress += 1
        
        
        f = open(self.progress_file, "w")
        f.write("%d" % self.progress)
        f.close()
        
    def select_action(self, state):
        if not hasattr(self, 'network'):
            self.setup_ann(state)
        q_values = self.network_output(numpy.reshape(state, (1, state.size)))[0]
        return self.exploration(q_values)

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
            elif key == "min_temperature":
                self.min_temperature = float(value)
            elif key == "temperature":
                self.temperature = self.init_temperature = float(value)
            elif key == "max_replay_size":
                self.max_replay_size = int(value)
            elif key in self.__dict__:
                self.__dict__[key] = float(value)
            elif not key in self.__dict__:
                raise Exception("Unknown setting: %s = %s" % (key, repr(value)))
            
    def set_test(self):
        self.test_performance = test
        self.test_count = 0
        if test:
            rew_file = os.path.basename(self.reward_file)
            rew_dir = os.path.dirname(self.reward_file)
            rew_file = os.path.join(rew_dir, "test_" + rew_file)

            prog_file = os.path.basename(self.progress_file)
            prog_dir = os.path.dirname(self.progress_file)
            prog_file = os.path.join(prog_dir, "test_" + prog_file)

            open(rew_file, 'a').write("------ NEW TEST ------\n")
            open(prog_file, 'a').write("------ NEW TEST ------\n")
            
    def setup_ann(self, nn_input, force_new = False):
        """
        This function loads existing networks or creates new networks for the
        actor and the critic.
        """
        #activation_func = libfann.SIGMOID_SYMMETRIC
        activation_func = self.activation_function

        try:
            num_inputs = int(nn_input)
        except:
            num_inputs = len(nn_input)
        
        self.x = T.matrix('x')
        
        network_weights = None
        if not force_new:
            network_weights = self.load_network(self.network_file)
        
        
        rng = numpy.random.RandomState()
        
        W3 = numpy.asarray(rng.uniform(
                            low=80,
                            high=80,
                            size=(num_inputs, self.num_outputs)), dtype=theano.config.floatX)  # @UndefinedVariable
                                
        b3 = numpy.zeros((self.num_outputs,), dtype=theano.config.floatX)
        
        if network_weights == None:
            network_weights = [W3, b3]
        
        self.network = linear_regression.NNRegression(input_data=self.x, n_in=num_inputs, 
                                                              n_out=self.num_outputs, weights = network_weights ,act_func = activation_func)
            
        self.network_output = theano.function([self.x],
                    outputs=self.network.output, allow_input_downcast=True)
        
                #cost = self.network.cost(self.y)
        self.ox = T.fvector('ox')
        self.oy = T.fvector('oy')
        
        cost = self.network.online_cost(self.ox, self.oy)
         
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = []
        for param in self.network.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)
            

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
        updates = []
        for param, gparam in zip(self.network.params, gparams):
            updates.append((param, param - self.learning_rate * gparam))
    
        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`

        self.train_network = theano.function(inputs=[self.ox, self.oy], outputs=cost,
                updates=updates, allow_input_downcast=True)
        
        ###BATCH NETWORK########
        self.y = T.fmatrix('y')
        index = T.lscalar('index')
        
            
        cost = self.network.cost(self.y)
             
            # compute the gradient of cost with respect to theta (sotred in params)
            # the resulting gradients will be stored in a list gparams
        gparams = []
        for param in self.network.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)    
            # specify how to update the parameters of the model as a list of
            # (variable, update expression) pairs
        updates = []
        for param, gparam in zip(self.network.params, gparams):
            updates.append((param, param - self.learning_rate * gparam))
        
            # compiling a Theano function `train_model` that returns the cost, but
            # in the same time updates the parameter of the model based on the rules
            # defined in `updates`
        self.train_batch = theano.function(inputs=[index], outputs=cost,
                    updates=updates,
                    givens={
                        self.x: self.shared_data[index * self.batch_size: (index + 1) * self.batch_size],
                        self.y: self.shared_labels[index * self.batch_size: (index + 1) * self.batch_size]})
    
    def update_networks(self, current_size = None, list = None, learning_rate = None):
        #print "Getting Values"
        
        if current_size:
            size = min(current_size, self.batch_size)
        else:
            size = self.batch_size
        f = open(self.weights_file, 'a')
        a = self.network.params[0].get_value()
        
        numpy.savetxt(f, a, fmt='%8.2f', delimiter=',', header="\n---BEGIN---\n", footer="\n----END----\n")
        #print "WEIGHTS-> Big Values:", a[numpy.where( a > 300)], " Max Value:", nump6y.max(a), " Min Value:" ,numpy.min(a)
        
        
        data, label = self.calculate_updates(list)

        self.y = T.fmatrix('y')
        index = T.lscalar('index')
        
        #cost = self.network.cost(self.y)
        L1_reg = 0.00
        L2_reg = 0.0001
        
        cost = self.network.cost(self.y) \
         + L1_reg * self.network.L1 \
         + L2_reg * self.network.L2_sqr
         
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = []
        for param in self.network.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
        updates = []
        for param, gparam in zip(self.network.params, gparams):
            if learning_rate:
                updates.append((param, param - learning_rate * gparam))
            else:
                updates.append((param, param - self.learning_rate * gparam))
    
        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`

        self.train_network = theano.function(inputs=[index], outputs=cost,
                updates=updates,
                givens={
                    self.x: data[index * size: (index + 1) * size],
                    self.y: label[index * size: (index + 1) * size]}
                                             )
        
        
        self.n_train_batches = data.get_value(borrow=True).shape[0] / size
        
        #for param in self.network.params:
        #    print "old"
        #    print param.get_value()
        
        #for minibatch_index in xrange(self.n_train_batches):
        #        minibatch_avg_cost = self.train_network(minibatch_index)
        
        average_error = 0
        last_error = 0
        epochs = 0
        while True:
            average_error = 0
            epochs += 1
            for minibatch_index in xrange(self.n_train_batches):
                minibatch_avg_cost = self.train_network(minibatch_index)
                average_error += minibatch_avg_cost
                
            average_error /= (self.n_train_batches)
            
            if math.fabs(average_error) < 0.001 or epochs > 50:
               #print "Average Batch Error: %3.6f -- %3.6f" % (average_error, last_error)
               print epochs
               break
            else:
                last_error = average_error
                #print "Average Batch Error:", average_error
        #for param in self.network.params:
        #    print "new"
        #    print param.get_value()
        
    def update_networks_plus(self, learning_rate = None, experience_replay = False):
        #print "Getting Values"
        
        self.rl_replay_data = self.rl_replay_data[-self.max_replay_size:]
        print "Replay Data Length", len(self.rl_replay_data)
        size = self.batch_size
        
        f = open(self.weights_file,'a')
        
        total_average_error = 0
        average_error = 0
        last_error = 0
        epochs = 0
        state = aciton = reward = new_state = None
        
        if experience_replay:
            shuffle(self.rl_replay_data)    
            state, action, reward, new_state = zip(*self.rl_replay_data)
            
            state = numpy.asarray(state, dtype = T.config.floatX)
            new_state = numpy.asarray(list(new_state), dtype = T.config.floatX)
            reward = numpy.asarray(reward, dtype = T.config.floatX)
            
    
        average_error = 0
        epochs += 1
        
        a = self.network.params[0].get_value()
        numpy.savetxt(f, a, fmt='%8.2f', delimiter=',', header="\n---BEGIN---\n", footer="\n----END----\n")
        #print "WEIGHTS-> Big Values:", a[numpy.where( a > 300)], " Max Value:", nump6y.max(a), " Min Value:" ,numpy.min(a)
        
        if experience_replay: 
                    
            current_q_values = self.network_output(state)
            next_q_values = self.network_output(new_state)
            temp_updates = reward + self.discount_factor * numpy.max(next_q_values, axis=1)
            #WARNING: I am assuming max reward is 100, so I replace all temp updates with 100 when tthe reward is that
            reached_goal_indices = numpy.where(reward >= 100)
            temp_updates[reached_goal_indices] = 100
            
            for idx, ac in enumerate(action):
                current_q_values[idx, ac] = temp_updates[idx]
            
            labels = numpy.asarray(current_q_values, dtype = theano.config.floatX)
            self.shared_data.set_value(state)
            self.shared_labels.set_value(labels)
        else:
            data = numpy.asarray(self.rl_data, dtype = theano.config.floatX)
            labels = numpy.asarray(self.rl_labels, dtype = theano.config.floatX)
            self.shared_data.set_value(data)
            self.shared_labels.set_value(labels)

        
        self.n_train_batches = self.shared_data.get_value(borrow=True).shape[0] / size
        
        for minibatch_index in xrange(self.n_train_batches):
            minibatch_avg_cost = self.train_batch(minibatch_index)
            average_error += minibatch_avg_cost
            
        if self.n_train_batches != 0:
            average_error /= (self.n_train_batches)
        
        #if math.fabs(average_error) < 0.00001 or epochs >= 50:
        #   print "Average Batch Error: %3.6f -- %3.6f" % (average_error, last_error)
        #   break
        else:
            last_error = average_error
            #print "Average Batch Error:", average_error
            
        if not experience_replay:
            #break
            pass
        
           
        self.rl_data = []
        self.rl_labels = []
        return average_error    

        
    def update_online(self, state, action, reward, next_state):
        if not hasattr(self, 'network'):
            self.setup_ann(state)
            
        old_values = self.network_output(numpy.reshape(state, (1, state.size)))[0]

        if next_state is None:
            update = reward
        else:
            values = self.network_output(numpy.reshape(next_state, (1, next_state.size)))[0]
            update = reward + self.discount_factor * numpy.max(values)
            
        old_values[action] = update
        
        state = numpy.asarray(state, T.config.floatX)
        new_values = numpy.asarray(old_values, dtype = T.config.floatX)
        
        
        self.train_network(state, new_values)
        
        self.rl_replay_data.append((state, action, reward, next_state))
        self.total_reward += (self.discount_factor ** self.steps) *  reward
        self.cum_reward += reward
        self.total_actions += 1

        self.steps += 1
    
    def clear_memory(self):
        self.rl_data = []
        self.rl_labels = []
        self.rl_replay_data = []
    def update_batch(self, state, action, reward, next_state):
        if not hasattr(self, 'network'):
            self.setup_ann(state)
            
        old_values = self.network_output(numpy.reshape(state, (1, state.size)))[0]

        if numpy.sum(next_state) < 1:
            update = reward
        else:
            values = self.network_output(numpy.reshape(next_state, (1, next_state.size)))[0]
            update = reward + self.discount_factor * numpy.max(values)
            
        old_values[action] = update
        
        self.rl_data.append(state)
        self.rl_labels.append(old_values)
        self.rl_replay_data.append((state, action, reward, next_state))
        
        self.total_reward += (self.discount_factor ** self.steps) *  reward
        self.cum_reward += reward
        self.total_actions += 1

        self.steps += 1
