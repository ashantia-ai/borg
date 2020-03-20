import os
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm 


def show_all_goals(data):

    plt.figure(2)
    colors = cm.rainbow(np.linspace(0, 1, len(data)))
    order = [0,8,2,7,1,5,6,3,4,9]

    for tmp_n, c in zip(order, colors):

        tmp = data['goal_'+str(tmp_n)]
        fin = []
        for i in tmp: fin.extend(i)
        fin = np.asarray(fin)
        plt.scatter(fin[:,0],fin[:,1], color = c)

    plt.ylim([0,100])
    plt.show()





def show_single_goal(data):

    plt.figure(1)
    colors = cm.rainbow(np.linspace(0, 1, len(data)))

    for tmp, c in zip(data, colors):

        single_data = np.asarray(tmp)
        plt.scatter(single_data[:,0], single_data[:,1], color = c)

    plt.show()


def main():

    starting_point = np.load('./starts.npy')
    order = [0,7,9,1,3,8,6,4,5,2]  # This is the order in which the NN trained. 
                                   # To see the order look at  time of modification of the folders
    all_data = {}
    for n_goal in order:
        
        curr_path = './goal_{}/'.format(n_goal)
        data_goal = []
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
    








if __name__ == '__main__':
    main()
