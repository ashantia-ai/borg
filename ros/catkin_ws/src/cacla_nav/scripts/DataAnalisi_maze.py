import os
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import pickle as pk
import pandas as p
import xlsxwriter

'''
This file is used for extract results for the Maze RL experiments of small kitchen and Cafe room. It will create a numpy file and a picture of numfer
of trials versus number of actions to raech the goal
'''

def show_all_goals(data, ind, goal_order):

    full_fig = None
    plt.figure(2)
    colors = cm.rainbow(np.linspace(0, 1, len(data))) #prepares rainbow color for coloring data points for different goals
    total_count = 0
    print goal_order[0]
    print zip(goal_order[0], colors)
    for tmp_n, c in zip(goal_order[0], colors):
        tmp = data['goal_{}_{}'.format(ind,tmp_n)]
        fin = []
        for i in tmp:
        
            i = np.asarray(i)
            tmp_counter = [total_count + j for j in xrange(len(i))] #create a new index 
            tmp_val = zip(tmp_counter, i[:,1]) #remove the index in the data and use the new one
            #full_fig.extend(zip(tmp_counter, i[:,1], color))
            fin.extend(tmp_val)
            total_count = tmp_counter[-1] # keep track of the total index
            
        fin = np.asarray(fin)
        
        
        
        #color = np.asarray([c] * fin.shape[0])
        
        goal = np.ones((fin.shape[0], 1)) * tmp_n
        
        res = np.hstack((fin, goal))
        
        
        
        if full_fig is not None:
            full_fig = np.vstack((full_fig, res))
            
        else:
            full_fig = res
        
        plt.scatter(fin[:,0],fin[:,1], color = c)

    print full_fig
    np.save("maze_res.npy", full_fig)
    writer = p.ExcelWriter('result.xlsx', engine='xlsxwriter')
    p.DataFrame(np.asarray(full_fig)).to_excel(writer, 'result')
    plt.ylim([0,200])
    plt.show()





def show_single_goal(data):

    plt.figure(1)
    colors = cm.rainbow(np.linspace(0, 1, len(data)))

    for tmp, c in zip(data, colors):

        single_data = np.asarray(tmp)
        plt.scatter(single_data[:,0], single_data[:,1], color = c)

    plt.show()

'''
This file read the output of maze "normal" (DOUBLE CHECK NORMAL) and multigoal and creates a combined figure out of it.
'''
def main():

    maze_ind = 0
    
    path = '/home/borg/amir-nav-experiments/maze_files/caffe_room/'
    starting_point = np.load(path + 'Maze_dataset/starts_{}.npy'.format(maze_ind))
    goals = np.load(path + 'Maze_dataset/goal_{}.npy'.format(maze_ind)).tolist()
    
    goal_order = np.load(path + 'Maze_dataset/goal_order.npy')
    ris_path = path + 'Multy_Maze_results/'
    saved_data = '/home/ashantia/University/experiments/maze/small_kitchen/Maze_results_new/single_goal_ris'
    num_goals = len(goals)
    num_start = len(starting_point)
    num_mazes = 1

    all_data = {}
    data_goal = []
    '''
    for maze_ind in xrange(num_mazes):
        for ind in xrange(num_start * num_goals):
            
            path = ris_path+'Maze_{}/'.format(ind+100*maze_ind)
            tmp_data = np.load(path+'fig2.npy')
            data_goal.append(tmp_data.tolist())
            if ind != 0 and (ind+1)%10 == 0:
            
                all_data['goal_{}_{}'.format(maze_ind,int((ind+1)/10)-1)] = data_goal
                data_goal = []
                
        show_all_goals(all_data, maze_ind)
    '''
    
    for maze_ind in xrange(num_mazes):
        for goals in xrange(num_goals):
            for starts in xrange(num_start):
            
                path = ris_path+'Maze_{}_{}/'.format(maze_ind, goals * 10 + starts)
                tmp_data = np.load(path+'fig2.npy')
                data_goal.append(tmp_data.tolist())
            
            all_data['goal_{}_{}'.format(maze_ind, goals)] = data_goal
            data_goal = []
                
    show_all_goals(all_data, maze_ind, goal_order)
    '''
    with open(saved_data,'w') as f:
        pk.dump(all_data, f)
        prev_data = 0

        for x,y in starting_point:

            path = curr_path + '{}_{}/'.format(x,y)
            tmp_data = np.load(path+'fig2.npy')
            tmp_data = tmp_data[prev_data:]
            new = [[int(x),int(y)] for x,y in tmp_data]
            prev_data += len(new)
            data_goal.append(new)

        all_data['goal_'+str(n_goal)] = data_goal

    print all_data

    #for g in  all_data.keys():

    show_all_goals(all_data)    

    '''
    








if __name__ == '__main__':
    main()
