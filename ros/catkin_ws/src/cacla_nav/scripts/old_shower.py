import rospy
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
def remove_empty_vert(grid):
    
    new=[] 
    for col in grid:
        if col.max()>0: new.append(col)
    return new


def gen_goals(maze, n_goal, n_start):
    x_pic, y_pic= maze.shape
    x_max, y_max = (16,16)
    goals=[]
    ind=[]
    
    start_y = 0.65
    start_x = -0.5
    #step_size_x = 5.5 / y_max
    #step_size_y = 4.3 / x_max
    step_size_x = step_size_y = 0.4
    
    while len(goals)<n_goal+n_start:
        x=random.randint(0, x_pic - 1)
        y=random.randint(0, y_pic - 1)
        if maze[x,y]==0:
            if (x,y) not in goals:
                ind.append((x,y))
                x2=(y*step_size_y) + 0.2
                y2=(-x*step_size_x) - 0.2
                #y2 = -x*step_size_x - start_x
                #x2 = -y*step_size_y - start_y
                
                goals.append((x2,y2))
    starts = goals[n_goal:]
    goals = goals[:n_goal]
    return (goals, starts, ind)

def remove_empty(grid):
    tmp=remove_empty_vert(grid)
    tmp=np.rot90(tmp, k=1)
    tmp=remove_empty_vert(tmp)
    tmp=np.rot90(tmp, k=3)
    return tmp
    

def read_av_section(cell,corner, l1,l2): #corner=(1,1) point to start, l1,l2 rectangular spec
    reg=cell[corner[0]-l1:corner[0]][:,corner[1]-l2:corner[1]]
    #print [corner[0]-l1,corner[0],corner[1]-l2,corner[1]]
    #print reg
    #print cell.shape
    return reg.max()
    

def make_grid(real_map, grid_shape=(11,11), approx=None):
    real_map=np.asarray(real_map)
    curr_shape=real_map.shape

    if approx!=None:
        grid_shape=(int(curr_shape[0]/approx),int(curr_shape[1]/approx)) 

    
    print curr_shape
    col_range=curr_shape[1]/grid_shape[1]
    row_range=curr_shape[0]/grid_shape[0]
    
   
    grid=np.zeros(grid_shape)
    
    cnt1=1
    cnt2=1
    cnt3=curr_shape[0]
    cnt4=curr_shape[1]
    for ind,row in enumerate(grid):
        ind=curr_shape[0]-ind
        cnt2=1
        cnt4=curr_shape[1]
        for ind2,cell in enumerate(row):
            ind2=curr_shape[1]-ind2
            point=(cnt3, cnt4)
            grid[grid_shape[0]-cnt1,grid_shape[1]-cnt2]=read_av_section(real_map,point, row_range,col_range)
            cnt2+=1
            cnt4-=col_range
        cnt1+=1
        cnt3-=row_range
    
    return grid


def check_closed_room_in_maze(maze, x=0, y=0): # x,y are the starting point
    tmp_maze=copy.deepcopy(maze)
    fin=copy.deepcopy(maze)
    tmp_maze[x,y]=2
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
           
pic=os.environ['BORG'] + '/ros/catkin_ws/src/sudo_stage/g_world/cherry.pgm'
save_path = os.environ['BORG'] + '/rl_nav_starts/'
image=cv2.imread(pic,0)
a=image/255.    
ind=a[:,:]>0.5
ind2=a[:,:]<=0.5
a[ind]=0
a[ind2]=1  



#a=np.rot90(a,k=3)
  
#a=[[random.randint(0,1) for i in xrange(10)] for j in xrange(10)]

b= a

b=b/b.max()    
ind=b[:,:]>0.5
ind2=b[:,:]<=0.5
b[ind]=1
b[ind2]=0 

b=make_grid(b,(16,16), None)
b=remove_empty(b)

c=check_closed_room_in_maze(b, x=2, y=2)

goals, starts, indeces=gen_goals(c, 10, 10) #in the function you need to set the origin!!!!!

#for x,y in goals: c[x,y]=2
 

np.save(save_path+'approx', c)
np.save(save_path+'goals', np.asarray(goals))
np.save(save_path+'starts', np.asarray(starts))
np.save(save_path+'indeces', np.asarray(indeces))

for i in xrange(10):
    c[indeces[i][0], indeces[i][1]] = 0.5
tmp = cv2.resize(c, (image.shape[0], image.shape[1]))
print np.max(image)
print np.min(image)
print np.max(tmp)
print np.min(tmp)
cv2.imwrite(save_path+'first_start.png', (tmp * 255 + image) / 2)

'''
for (x,y),(ind1,ind2) in zip(goals, indeces):
    tmp = copy.deepcopy(c)
    tmp2 = copy.deepcopy(c)
    tmp[x,y]=2
    tmp2[ind1,ind2]=2
    plt.matshow(tmp,2)
    plt.matshow(tmp2,3)
    plt.show() 

plt.matshow(a,1)
plt.matshow(b,2)
plt.matshow(c,3)
plt.show()           
'''
