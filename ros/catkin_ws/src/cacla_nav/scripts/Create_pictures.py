import numpy as np  
import matplotlib.pyplot as plt
import os
import cv2

def do_pic(base_path, path, maz_path):
    #curr_goal = maz_path.split('_')[-1]
    #curr_goal = int(curr_goal[:-1])
    #maze=np.load(base_path+'approx.npy')
    fig1=np.load(path+'fig1.npy')
    fig2=np.load(path+'fig2.npy')

    if len(fig1) >0:
        f1=plt.figure(1)
        plt.scatter(fig1[:,0],fig1[:,1])
        f1.savefig(path+'fig1.png')
        plt.close()

    if len(fig2) >0:
        f2=plt.figure(2)
        plt.scatter(fig2[:,0],fig2[:,1])
        f2.savefig(path+'fig2.png')
        plt.close()

    #f3=plt.figure(3)
    #plt.matshow(maze, fignum=3)
    #f3.savefig(path+'maze.png')
    #plt.close()


if __name__=='__main__':
    path='/home/amir/sudo/ros/catkin_ws/src/cacla_nav/goals_starts/'
    goals_to_do = [2,9]
    #mazes=os.listdir(path)
    #tot_pic=len(mazes)

    for goal in goals_to_do:

        curr_path = path+'goal_{}/'.format(goal)
        starts = os.listdir(curr_path)
        starts.remove('nfq')
        starts.remove('maze.png')

        for st in starts:
            
            fin_path = curr_path+st+'/'
            do_pic(path,fin_path, curr_path)

       


