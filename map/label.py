#!/usr/bin/env python
import sys
import os 
import numpy as np
import cv2
from tqdm import tqdm

import yaml
    
        
if __name__ == '__main__':
    # x_norm = 4.75
    # y_norm = 10.25

    train_path = os.listdir("data/train_set/")
    test_path = os.listdir("data/test_set/")
    eval_path = os.listdir("data/eval_set/")

    train_dir = "data/train_set"
    test_dir = "data/test_set"
    eval_dir = "data/eval_set"

    if not os.path.exists(train_dir):
                    os.makedirs(train_dir)
    if not os.path.exists(test_dir):
                    os.makedirs(test_dir)
    if not os.path.exists(eval_dir):
                    os.makedirs(eval_dir)

    train_size = 470000
    test_size = 310000
    eval_size = 150000






    images = np.empty((train_size,84,84,3), dtype=np.uint8)
    labels = np.empty((train_size,4))
    idx = 0
    for num_file, data in (enumerate(train_path)):
        with open("data/train_set/{}".format(data), 'r') as stream:
             print data
             items = yaml.load(stream)
             # items = items[0:-1:5]
             for item in items:
                 print item
                 cvim = cv2.imread(item["image"]["file"])
                 cvim = cv2.cvtColor(cvim, cv2.COLOR_BGR2RGB)
                 print "here"
                 print cvim.shape
                 cvim = cv2.resize(cvim,(84,84))
                 image = np.asarray(cvim)
                 image = image.reshape(84, 84, 3)
                 images[idx] = image
                 location = item["image"]["location"]
                 labels[idx] = np.array([location["x"], location["y"], item["image"]["sin"], item["image"]["cos"]])
                 idx +=1
    np.save("{}/images.npy".format(train_dir), images)
    np.save("{}/labels.npy".format(train_dir), labels)



    images = np.empty((test_size,84,84,3), dtype=np.uint8)
    labels = np.empty((test_size,4))
    idx = 0
    for num_file, data in tqdm(enumerate(test_path)):
        with open("data/test_set/{}".format(data), 'r') as stream:
            items = yaml.load(stream)
            # items = items[0:-1:5]
            for item in items:
                cvim = cv2.imread(item["image"]["file"])
                #cvim = cv2.cvtColor(cvim, cv2.    COLOR_BGR2RGB)
                cvim = cv2.resize(cvim,(84,84))
                image = np.asarray(cvim)
                image = image.reshape(84, 84, 3)
                images[idx] = image
                location = item["image"]["location"]
                labels[idx] = np.array([location["x"], location["y"], item["image"]["sin"], item["image"]["cos"]])
                idx +=1
    np.save("{}/images.npy".format(test_dir), images)
    np.save("{}/labels.npy".format(test_dir), labels)



    images = np.empty((eval_size,84,84,3), dtype=np.uint8)
    labels = np.empty((eval_size,4))
    idx = 0
    for num_file, data in tqdm(enumerate(eval_path)):
        with open("data/eval_set/{}".format(data), 'r') as stream:
            items = yaml.load(stream)
            # items = items[0:-1:5]
            for item in items:
                cvim = cv2.imread(item["image"]["file"])
                #cvim = cv2.cvtColor(cvim, cv2.    COLOR_BGR2RGB)
                cvim = cv2.resize(cvim,(84,84))
                image = np.asarray(cvim)
                image = image.reshape(84, 84, 3)
                images[idx] = image
                location = item["image"]["location"]
                labels[idx] = np.array([location["x"], location["y"], item["image"]["sin"], item["image"]["cos"]])
                idx +=1
    np.save("{}/images.npy".format(eval_dir), images)
    np.save("{}/labels.npy".format(eval_dir), labels)




# awk '{filename = "data_set/data_set" int((NR-1)/70000) ".yaml"; print >> filename}' data_set.yaml 
