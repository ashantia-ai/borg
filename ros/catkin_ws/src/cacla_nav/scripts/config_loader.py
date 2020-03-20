'''
Created on Nov 18, 2014

@author: shantia
'''
import time
import sys
import getopt
import logging
import os
from string import upper
import itertools

import multiprocessing
from multiprocessing import Lock
import subprocess
import xmlrpclib
import math
# Brain imports

import configparse
from nose.plugins import multiprocess

import util.ticker

class RPC(object):
    def __init__(self, address_ranges):
        self.address_ranges = []
        
        for addr in address_ranges:
            self.address_ranges.append((addr, True, None)) 
        
        self.proxies = []
        
        self.finished_indices = []
    
    def check_finished(self):
        finished_processes = 0
        for index, item in enumerate(self.address_ranges):
            addr, free, proxy = item
            if not free:
                if proxy.is_finished():
                    self.address_ranges[index] = (addr, True, proxy)
                    finished_processes+= 1
                    
        return finished_processes
                
    
    def remote_call(self, params, server):
        proxy = xmlrpclib.ServerProxy(server)
        
        try:
            if proxy.start(params):
                print "Succesfully started"
            
        except xmlrpclib.ProtocolError as err:
            print "A protocol error occurred"
            print "URL: %s" % err.url
            print "HTTP/HTTPS headers: %s" % err.headers
            print "Error code: %d" % err.errcode
            print "Error message: %s" % err.errmsg
        except xmlrpclib.Error as err:
            print err
    
        return proxy
    
    def new_experiment(self, params):
        for index, item in enumerate(self.address_ranges):
            addr, free, _ = item
            if free:
                proxy = self.remote_call(params, addr)
                self.address_ranges[index] = (addr, False, proxy)
                return
            
        raise 'No Free PCs Available'
    
def check_options(option_dict):
    """
    Check whether required options are set (either on commandline or
    in configuration file).
    Throws an exception if one or more configuration options is invalid.
    """
    #validate nao configuration:
    features = option_dict.get_option('navigation', 'cacla', 'remote')
    if features == None:
        raise Exception("Features are not specified!")

    
def parse_args(sysargs):
    """
    Parse command line arguments
    """
    optlist, args = getopt.getopt(sysargs[1:], 'hb:', 
        ['nao_ip=','no_nao','pioneer_ip=','no_pioneer','nao_port=','pioneer_port=',
        'kinect_ip=','communicator_port=','speech_port=','starting_behavior=','brain_speed=',
        'log=','log-level=','log-file=','log-format=','help','profile'])
    print optlist, args
    option_dict = configparse.ParameterDict()

    for opt, value in optlist:
        if opt in ("-h", "--help"):
            sys.exit()
        elif opt in ("--feature"):
            option_dict.add_option('feature', 'features', value)
        elif opt in ("--communicator_port"):
            option_dict.add_option('vision_controller', 'communicator_port', value)

    return option_dict, args

def load_config(sysargs):
    """
    Load configuration file and combine this with command line arguments
    """
    if len(sysargs) < 2:
        sys.exit()
    option_dict, args = parse_args(sysargs)
    
    if len(args) >= 1:
        config_file = args[0]
        configparse.parse_config(config_file, option_dict) #does not overwrite existing arguments in option_dict

    try:
        check_options(option_dict)
    except Exception as e:
        print e
        sys.exit()
        
    return option_dict
    
def CACLA_training():
    print "arguments: ", sys.argv
    # Set this to True to enable profiling output of the brain, or use
    # --profile command line argument
    profile = False
    # python fannLearning.py feature hiddenUnits learningRate
    
    parameter_dict = load_config(sys.argv)
    
    #Cacla navigation parameters
    loop_rate_list = parameter_dict.get_option('navigation', 'loop_rate')
    loop_rates = loop_rate_list.split(",");
    initial_position_values = parameter_dict.get_option('navigation', 'initial_position')
    initial_positions = initial_position_values.split(":")
    
    goal_position_values = parameter_dict.get_option('navigation', 'goal_position')
    goal_positions = goal_position_values.split(":")
    
    input_type_list = parameter_dict.get_option('navigation', 'input_type')
    input_types = input_type_list.split(",");
    
    trial_length_list = parameter_dict.get_option('navigation', 'trial_length')
    trial_lengths = trial_length_list.split(",");
    trial_lengths  = [float(x) for x in trial_lengths]
    
    location_fix_reward_list = parameter_dict.get_option('navigation', 'location_fix_reward')
    location_fix_rewards = location_fix_reward_list.split(",");
    location_fix_rewards = [float(x) for x in location_fix_rewards]
    
    angular_fix_reward_list = parameter_dict.get_option('navigation', 'angular_fix_reward')
    angular_fix_rewards = angular_fix_reward_list.split(",");
    angular_fix_rewards = [float(x) for x in angular_fix_rewards]
        
    complex_reward_list = parameter_dict.get_option('navigation', 'complex_reward')
    complex_rewards = complex_reward_list.split(",");
    complex_rewards = [bool(x) for x in complex_rewards]
    
    fix_punishment_list = parameter_dict.get_option('navigation', 'fix_punishment')
    fix_punishments = fix_punishment_list.split(",");
    fix_punishments = [float(x) for x in fix_punishments]
    
    trial_fail_punishment_list = parameter_dict.get_option('navigation', 'trial_fail_punishment')
    trial_fail_punishments = trial_fail_punishment_list.split(",");
    trial_fail_punishments = [float(x) for x in trial_fail_punishments ]
    
    obstacle_punishment_list = parameter_dict.get_option('navigation', 'obstacle_punishment')
    obstacle_punishments = obstacle_punishment_list.split(",");
    obstacle_punishments = [float(x) for x in obstacle_punishments]
    
    negative_speed_punishment_list = parameter_dict.get_option('navigation', 'negative_speed_punishment')
    negative_speed_punishments = negative_speed_punishment_list.split(",");
    negative_speed_punishments = [float(x) for x in negative_speed_punishments]
    
    queue_size_list = parameter_dict.get_option('navigation', 'queue_size')
    queue_sizes = queue_size_list.split(",");
    queue_sizes = [float(x) for x in queue_sizes]
    
    acceptable_run_threshold_list = parameter_dict.get_option('navigation', 'acceptable_run_threshold')
    acceptable_run_thresholds = acceptable_run_threshold_list.split(",");
    acceptable_run_thresholds = [float(x) for x in acceptable_run_thresholds]
    
    success_trials_threshold_list = parameter_dict.get_option('navigation', 'success_trials_threshold')
    success_trials_thresholds = success_trials_threshold_list.split(",");
    success_trials_thresholds = [float(x) for x in success_trials_thresholds]
    
    yaw_change_list = parameter_dict.get_option('navigation', 'yaw_change')
    yaw_changes = yaw_change_list.split(",");
    yaw_changes = [float(x) for x in yaw_changes]
    
    x_change_list = parameter_dict.get_option('navigation', 'x_change')
    x_changes = x_change_list.split(",");
    x_changes = [float(x) for x in x_changes]
    
    y_change_list = parameter_dict.get_option('navigation', 'y_change')
    y_changes = y_change_list.split(",");
    y_changes = [float(x) for x in y_changes ]
    
    replay_frequency_list = parameter_dict.get_option('navigation', 'replay_frequency')
    replay_frequencies = replay_frequency_list.split(",");
    replay_frequencies = [float(x) for x in replay_frequencies]
    

    
    
    
    #CACLA algorithm parameters
    num_hiddens_list = parameter_dict.get_option('cacla', 'num_hiddens')
    num_hiddens = num_hiddens_list.split(",")
    num_hiddens = [float(x) for x in num_hiddens]
    
    activation_function_list = parameter_dict.get_option('cacla', 'activation_function')
    activation_functions = activation_function_list.split(",");
    activation_functions = [float(x) for x in activation_functions]
    
    
    alpha_value_list = parameter_dict.get_option('cacla', 'alpha')
    alpha_values_temp = alpha_value_list.split(",")
    alpha_values = []
    for alpha in alpha_values_temp:
        alpha_min, alpha_max = alpha.split(":")
        alpha_values.append((float(alpha_min),float(alpha_max)))
    
    beta_value_list = parameter_dict.get_option('cacla', 'beta')
    beta_values_temp = beta_value_list.split(",");
    beta_values = []
    for beta in beta_values_temp:
        beta_min, beta_max = beta.split(":")
        beta_values.append((float(beta_min),float(beta_max)))
        
    sigma_list = parameter_dict.get_option('cacla', 'sigma')
    sigma_values_temp = sigma_list.split(",")
    sigma_values = []
    for sigma in sigma_values_temp:
        sigma_min, sigma_max = sigma.split(":")
        sigma_values.append((float(sigma_min),float(sigma_max)))
    
    discount_factor_list = parameter_dict.get_option('cacla', 'discount_factor') 
    discount_factors = discount_factor_list.split(",")
    
    random_decay_list = parameter_dict.get_option('cacla', 'random_decay') 
    random_decay_values = random_decay_list.split(",")
    
    ex_strat_list = parameter_dict.get_option('cacla', 'exploration_strategy') 
    ex_strat_values = ex_strat_list.split(",")
    
    base_ip_list = parameter_dict.get_option('remote', 'base_ip')
    base_ips = base_ip_list.split(",")
    
    ip_range_list = parameter_dict.get_option('remote', 'ip_range')
    ip_ranges_temp= ip_range_list.split(",")
    
    address_ranges = []
    template_addr = "http://%s:8000/"
    for i in range(len(base_ips)):
        ip_ranges = ip_ranges_temp[i].split(":")
        for ip in ip_ranges:
            address_ranges.append(template_addr % (base_ips[i] + ip))
        

    
    #connecting features that should not be combined
    no_combined_set = []
    #for feature, input, output in zip(features, n_inputs, n_outputs):
    #    no_combined_set.append((feature,input,output))

    all_combinations = itertools.product(loop_rates, [tuple(initial_positions)], [tuple(goal_positions)], input_types 
                                         , alpha_values, beta_values, sigma_values, discount_factors, random_decay_values, ex_strat_values
                                         ,trial_lengths, location_fix_rewards, angular_fix_rewards, complex_rewards
                                         ,fix_punishments, trial_fail_punishments, obstacle_punishments, negative_speed_punishments
                                         ,queue_sizes, acceptable_run_thresholds, success_trials_thresholds
                                         , yaw_changes, x_changes,y_changes, replay_frequencies, num_hiddens, activation_functions)
    

    run_commands = []
    for combination in all_combinations:
        print combination
        #run_commands.append((combination[0], combination[1], combination[2], combination[3], combination[4], combination[5], combination[6]))
        run_commands.append(combination)
    
    completed_processes = 0
    
    print "Addresses to Connect:"
    print address_ranges
    remote_process = RPC(address_ranges)
    available_pcs = len(address_ranges)
    total_processes = len(run_commands)
    lock = Lock()
    
    sleep = util.ticker.Ticker(frequency = 1)
    
    while completed_processes != total_processes:
        sleep.tick()
        if available_pcs > 0 and run_commands:
            command = run_commands.pop()
            (loop_rate, init_pos, goal_pos, type, alpha, beta, sigma, discount_factor, random_decay, ex_strat, 
            trial_length, location_fix_reward, angular_fix_reward, complex_reward,
            fix_punishment, trial_fail_punishment, obstacle_punishment, negative_speed_punishment,
            queue_size, acceptable_run_threshold, success_trials_threshold,
            yaw_change, x_change,y_change, replay_frequency, num_hidden, activation_function) = command
            
            cacla_args = {'alpha':tuple(alpha), 'beta':tuple(beta), 'sigma':tuple(sigma), 'discount_factor':float(discount_factor),
                          'random_decay':float(random_decay), 'explore_strategy':int(ex_strat), 'num_hiddens':num_hidden,
                          'activation_function':activation_function
                          }
            
            dict = {'loop_rate':int(loop_rate), 'initial_position':init_pos, 'goal_position':goal_pos,'input_type':type, 'cacla_args':cacla_args,
                    'trial_length':trial_length, 'location_fix_reward':location_fix_reward, 'angular_fix_reward':angular_fix_reward, 
                    'complex_reward':complex_reward, 'fix_punishment':fix_punishment,
                    'trial_fail_punishment':trial_fail_punishment, 'obstacle_punishment':obstacle_punishment, 'negative_speed_punishment':negative_speed_punishment,
                    'queue_size':queue_size, 'acceptable_run_threshold':acceptable_run_threshold, 'success_trials_threshold':success_trials_threshold,
                    'yaw_change':yaw_change, 'x_change':x_change, 'y_change':y_change, 'replay_frequency':replay_frequency}
            
            temp_process = remote_process.new_experiment(dict)
    
            available_pcs -= 1
        else:
            kill_list = []
            
            for i in range(remote_process.check_finished()):
                
                completed_processes += 1
                available_pcs += 1
                print completed_processes, " processes are finished"
    print "All processes have finished"
    
    
if __name__ == "__main__":
    #NN_training()
    CACLA_training()
                    