import numpy as np
import pandas
'''
This script reads a saved neural network, and then puts the Q-values in a excel sheet.
The sizes for the maze is static and you have to set the row and col numbers.
The excel sheet will have multuple sheets, containing qvalues for all separate actions, and the best one.
'''
writer = pandas.ExcelWriter('q_values.xlsx', engine='xlsxwriter')
n = np.load("network.net")[0]

row_no = 47
col_no = 24

n0 = n[:,0].reshape(row_no, col_no)
n1 = n[:,1].reshape(row_no, col_no)
n2 = n[:,2].reshape(row_no, col_no)
n3 = n[:,3].reshape(row_no, col_no)

n0[n0 == 80] = 0
n1[n0 == 80] = 0
n2[n0 == 80] = 0
n3[n0 == 80] = 0

ns = (n0 + n1 + n2 + n3) / 4
nm = np.maximum.reduce([n0,n1,n2,n3])

height = n0.shape[0]
width = n0.shape[1]

best = np.chararray((height, width), itemsize=1, unicode=True)
max_q = np.zeros((height, width))
print n0.shape
print best.shape
for i in xrange(height):
    for j in xrange(width):
        if n0[i,j] == 80 or n0[i,j] == -10:
            best[i,j] = 'B'
            continue

        up = n0[i,j]
        left = n1[i,j]
        down = n2[i,j]
        right = n3[i,j]
        
        highest = max(up,left,down,right)
        if highest == up:
            best[i,j] = u'\u2191'
            max_q[i,j] = up
        elif highest == left:
            best[i,j] = u'\u2190'
            max_q[i,j] = left
        elif highest == down:
            best[i,j] = u'\u2193'
            max_q[i,j] = down
        elif highest == right:
            best[i,j] = u'\u2192'
            max_q[i,j] = right




print pandas.DataFrame(n0)
print '----------------'
print pandas.DataFrame(n1)
print '----------------'
print pandas.DataFrame(n2)
print '----------------'
print pandas.DataFrame(n3)
print '----------------'
print pandas.DataFrame(ns)
print '----------------'
print pandas.DataFrame(nm)
print '----------------'
print pandas.DataFrame(best)

pandas.DataFrame(nm).to_csv("./max.csv")
pandas.DataFrame(n0).to_excel(writer, 'up')
pandas.DataFrame(n1).to_excel(writer, 'left')
pandas.DataFrame(n2).to_excel(writer, 'down')
pandas.DataFrame(n3).to_excel(writer, 'right')
pandas.DataFrame(max_q).to_excel(writer, 'max')
pandas.DataFrame(best).to_excel(writer, 'best')

