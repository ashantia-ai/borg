import numpy as np
import os
import cPickle as cp

for root, dirs, files in os.walk('./'):
	for filename in files:
		if filename == 'network.net':
			net = np.load(os.path.join(root,filename))[0]
			print os.path.join(root, filename)
			net[net == 80] = -10
			net[net == 40] = -10
			f = open(os.path.join(root,filename), 'wb')
			cp.dump([net], f, protocol=cp.HIGHEST_PROTOCOL)
			f.close()

