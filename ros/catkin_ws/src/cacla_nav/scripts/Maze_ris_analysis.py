import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import re

'''
A Magical file that I don't know what it exactly does
It read all the fig2 files, and apparently tries to average it. but the results are weird.
'''

def tryint(s):
    try:
        return int(s)
    except:
        return s
    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)



#path = '/home/borg/Nav_NN/'
path = '/home/borg/amir-nav-experiments/maze_files/caffe_room/Multy_Maze_results/'
nav = os.walk(path)

mazes = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
sort_nicely(mazes)
print 'mazes: ', mazes

max_data = 0
data_saver = []
for folder in mazes:
    if folder[-4:]!='.png':
        tmp_path = path + folder + '/'
        data = np.load(tmp_path + 'fig2.npy')  
        data_saver.append(data)          
        if len(data) > max_data:
            max_data = len(data)

data = np.asarray(data_saver)
num = np.zeros(max_data)
graph = [[] for i in xrange(max_data)]

for ind in xrange(max_data):
    for tmp_data in data:
        if ind < len((tmp_data)-1):
            graph[ind].append(tmp_data[ind][1])
            num[ind] += 1  
              
high = []
low = []
av = []
            
for ind, i in enumerate(graph):
    avr = np.mean(i)
    std = np.std(i)
    
    av.append(avr)
    high.append(avr+std)
    if avr-std > 0:
        low.append(avr-std)
    else:
        low.append(0)
    
num = np.asarray(num)
ind = num[:]>1000
num[ind] = 1000
 
f1=plt.figure(1)
plt.plot(num)
#plt.plot(high)
plt.plot(av)
#plt.plot(low)
plt.show()
f1.savefig(path+'AvMaze.png')
plt.close()





    
