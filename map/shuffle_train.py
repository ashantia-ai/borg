import numpy as np
import glob

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

'''
Shuffles training data for the CNN training
'''
if __name__ == '__main__' :
    sets = glob.glob('data/train/labels*')
    sizes = []
    for dataset in sets:
        x = np.load(dataset)
        sizes.append(x.shape)
    x = np.load('data/train/images0.npy')
    y = np.load('data/train/labels0.npy')
    for idx in range(1, len(sets)):
        x = np.concatenate([x, np.load('data/train/images{}.npy'.format(idx))])
        y = np.concatenate([y, np.load('data/train/labels{}.npy'.format(idx))])
    
    shuffle_in_unison(x, y)
    total_size = 0
    for idx, size in enumerate(sizes):
        size = size[0]
        np.save("data/train/images{}.npy".format(idx), x[total_size:total_size+size])
        np.save("data/train/labels{}.npy".format(idx), y[total_size:total_size+size])
        total_size += size


