import numpy as np
import pandas

writer = pandas.ExcelWriter('result.xlsx', engine='xlsxwriter')
n = np.load("maze_res.npy")
pandas.DataFrame(n).to_excel(writer, 'result')

