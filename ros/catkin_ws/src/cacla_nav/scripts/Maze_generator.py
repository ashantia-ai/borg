import numpy as np
import numpy
import copy
import os
import random
import time
import matplotlib.pyplot as plt



class Maze_generator(object):
    def __init__(self, n_maze=1000, goal_per_maze=10, start_per_maze=10,
                wall_freq=0.3, min_goal_distance=7, maze_shape=(10,10), 
                save_path='/home/ashantia/University/experiments/random_maze/Maze_dataset/', random_start=False):

        self.n_maze = n_maze
        self.goal_per_maze = goal_per_maze
        self.start_per_maze = start_per_maze
        self.min_goal_distance = min_goal_distance
        self.maze_shape = maze_shape
        self.save_path = save_path
        self.wall_freq = wall_freq
        self.random_start = random_start

        self.name_counter=0

    '''
    Main function to call for maze generation.
    '''
    def generate_mazes(self): 
        cnt=0   
        while cnt<self.n_maze:
            maze=self.__generate_random_maze(self.maze_shape, self.wall_freq)
            goals,starts=self.__chose_random_goal_with_min_dist(maze, self.min_goal_distance, self.goal_per_maze, self.start_per_maze, self.random_start)
            if len(goals) >0:
            
                #self.__visualize_maze(maze)
                #time.sleep(0.5)
                np.save(self.save_path+'maze_'+str(cnt), maze)
                np.save(self.save_path+'goal_'+str(cnt), goals)
                if self.random_start:
                    np.save(self.save_path+'starts_'+str(cnt), starts)    
                cnt+=1


    '''
    This function creates random mazes by going through columns of the desired maze, and 
    setting obstacles with the given frequency. At the end, we fill inside all the obstacles if necessary.
    If the desired maze with the desired start/goal distances cannot be found, the maze is thrown away and a new one 
    is generated.
    '''        
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

    '''
    This function checks for obstacles which form a closed space, and fills them in.
    '''
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

    '''
    Visualizing the maze
    '''
    def __visualize_maze(self, maze):
        #plt.figure(1)
        plt.matshow(maze)
        plt.show(block=False)
        
    
        
        
                    
    def __chose_random_goal_with_min_dist(self, maze, min_dist=7, n_of_goals=10, n_of_start=10, add_random_start=False):
        
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

        if tot_goals<n_of_goals and not add_random_start:
        
            print 'This maze is too small, please create another'
            return ([],[])
            
        if tot_goals<n_of_goals+n_of_start and add_random_start:
        
            print 'This maze is too small for {} goals and {} starting points, please create another'.format(n_of_goals,n_of_start)
            return ([],[])

        goals_to_return=[]        
        for i in xrange(n_of_goals):
            
            random_goal_ind=random.randint(0, len(goals)-1)
            (x,y)=goals[random_goal_ind]
            goals_to_return.append((x,y))
            goals=goals[:random_goal_ind]+goals[random_goal_ind+1:] #remove selected goal
            
        if not add_random_start:
            
            return (np.asarray(goals_to_return),[])
        
        else:
            
            starts_to_return=[(0,0)] 
            for i in xrange(n_of_start-1):
                
                random_goal_ind=random.randint(0, len(goals)-1)
                (x,y)=goals[random_goal_ind]
                starts_to_return.append((x,y))
                goals=goals[:random_goal_ind]+goals[random_goal_ind+1:]
                
            return (np.asarray(goals_to_return), np.asarray(starts_to_return))    

'''
This Python main program generates random mazes of certain size.
@param n_maze: number of mazes to generate
@param goal_per_maze: The number of goals that you want for the reinforcement learning program
@param start_per_maze: The number of starting points that you want for the reinforcement learning program
@param wall_freq: The percentage of walls/obstacles.
@param min_goal_distance: The required minimum goal distance between starting point and the goal points
@param maze_shape: A tuple with the size of the maze
@param save_path: the path to save the maze dataset.
@param random_start: whether to randomize all starting points or only use 0,0 as start. Keep it True
'''
if __name__=='__main__':
    generator=Maze_generator(n_maze=10, goal_per_maze=10, start_per_maze=10,
                wall_freq=0.3, min_goal_distance=7, maze_shape=(11,11), 
                save_path='/home/ashantia/University/experiments/random_maze/Maze_dataset/', random_start=True)
    
    generator.generate_mazes()    


