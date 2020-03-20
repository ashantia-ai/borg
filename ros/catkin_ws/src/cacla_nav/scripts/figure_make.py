import matplotlib.pyplot as plt
from multiprocessing import Lock, Value, Array

fig = plt.figure()
fig1_h = fig.add_subplot(221)
fig1_h.set_xlim([0, 100000])
fig1_h.set_ylim([-5, 20])
fig2_h = fig.add_subplot(222)
fig2_h.set_xlim([0, 10000])
fig2_h.set_ylim([0, 20])

plt.show(block=False)


def update(lock, value, array):
    
    while True:
        if value.value == 0:
            print "updating fig1"
            print len(array)
            print array
            x, y = zip(*array)
            fig1_h.cla()
            fig1_h.plot(numpy.asarray(x),numpy.asarray(y), 'b-')
            
            '''
            x, y = zip(*self.fig2_data)
            self.fig2_h.cla()
            self.fig2_h.plot(numpy.asarray(x),numpy.asarray(y), 'b+')
            '''
            
            value.value = 1.0
            
def update2(lock, value, array):
    
    while True:
        if value.value == 0:
            
            print "updating fig2"
            print array
            x, y = zip(*array)
            fig2_h.cla()
            fig2_h.plot(numpy.asarray(x),numpy.asarray(y), 'b-')
            
            '''
            x, y = zip(*self.fig2_data)
            self.fig2_h.cla()
            self.fig2_h.plot(numpy.asarray(x),numpy.asarray(y), 'b+')
            '''
            
            value.value = 1.0