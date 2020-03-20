#!/usr/bin/env python
import sys
import os
from tqdm import tqdm
import yaml
import numpy as np
import math

import cv2
'''
This file goes over a recorded data set folder (With images and everything), and creates numpy train/test/eval set.
You have to provide correct min/max location of the map for proper normalization. In addition, you have to take care about the recorded angles and 
set the range correctl on line 54.
'''

if __name__ == '__main__':
	
	# random.seed(42)
	switch = 0
	
	#Cafe room coordinate. not sure whether these are the ones.
	#min_x = -3.5
	#max_x = 4.75
	#min_y= -10.25
	#max_y = 6

    #Kitchen coordinates. updated 16th July 2019
	min_x = -0.507
	max_x = 4.98
	min_y= -4.77
	max_y = 0.58

	mean_x = (min_x+max_x)/2
	mean_y = (min_y+max_y)/2

	max_abs = max(abs(max_x-mean_x), abs(max_y-mean_y))
	print(mean_x)
	print(mean_y)
	print(max_abs)
	
	train_idx=0
	test_idx=0
	eval_idx=0
	train_nr=0
	test_nr=0
	eval_nr=0

	train_images = np.empty((10000,84,84,3), dtype=np.uint8)
	train_labels = np.empty((10000,4))
	test_images = np.empty((10000,84,84,3), dtype=np.uint8)
	test_labels = np.empty((10000,4))
	eval_images = np.empty((10000,84,84,3), dtype=np.uint8)
	eval_labels = np.empty((10000,4))
	angle_range = np.arange(-180, 180, 10)

	with open("data_set.yaml", 'r') as stream:
		items = yaml.load(stream)
		for item in tqdm(items):
			image = cv2.imread(item["image"]["file"])
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = cv2.resize(image,(84,84))
			image = np.asarray(image)
			image = image.reshape(84, 84, 3)
			location = item["image"]["location"]
			sin = math.sin(math.radians(item["image"]["angle"]))
			cos = math.cos(math.radians(item["image"]["angle"]))
			label = np.array([(location["x"]-mean_x)/max_abs, (location["y"]-mean_y)/max_abs, sin, cos])
			if (np.max(label) > 1):
				print "WROOOOOOOOOOOOOOOOONG"
				exit()
			in_range = False
			for idx in angle_range:
				if item["image"]["angle"] >= idx and item["image"]["angle"] < idx+5.2:
					in_range = True

			if 	in_range:
				if train_idx==10000:
					if not os.path.exists("data/train"):
						os.makedirs("data/train")
					np.save("{}/images{}.npy".format("data/train", train_nr), train_images)
					np.save("{}/labels{}.npy".format("data/train", train_nr), train_labels)
					train_nr+=1
					train_idx=0
					train_images = np.empty((10000,84,84,3), dtype=np.uint8)
					train_labels = np.empty((10000,4))
				train_images[train_idx] = image
				train_labels[train_idx] = label
				train_idx+=1

			elif switch<2:
				
				if test_idx==10000:
					if not os.path.exists("data/test"):
						os.makedirs("data/test")
					np.save("{}/images{}.npy".format("data/test", test_nr), test_images)
					np.save("{}/labels{}.npy".format("data/test", test_nr), test_labels)
					test_nr+=1
					test_idx=0
					test_images = np.empty((10000,84,84,3), dtype=np.uint8)
					test_labels = np.empty((10000,4))
				test_images[test_idx] = image
				test_labels[test_idx] = label
				test_idx+=1
				switch += 1
			else:
				
				if eval_idx==10000:
					if not os.path.exists("data/eval"):
						os.makedirs("data/eval")
					np.save("{}/images{}.npy".format("data/eval", eval_nr), eval_images)
					np.save("{}/labels{}.npy".format("data/eval", eval_nr), eval_labels)
					eval_nr+=1
					eval_idx=0
					eval_images = np.empty((10000,84,84,3), dtype=np.uint8)
					eval_labels = np.empty((10000,4))
				eval_images[eval_idx] = image
				eval_labels[eval_idx] = label
				eval_idx+=1
				switch = 0
		if train_idx!=0:
			np.save("{}/images{}.npy".format("data/train", train_nr), train_images[:train_idx])
			np.save("{}/labels{}.npy".format("data/train", train_nr), train_labels[:train_idx])
		if test_idx!=0:
			np.save("{}/images{}.npy".format("data/test", test_nr), test_images[:test_idx])
			np.save("{}/labels{}.npy".format("data/test", test_nr), test_labels[:test_idx])
		if eval_idx!=0:
			np.save("{}/images{}.npy".format("data/eval", eval_nr), eval_images[:eval_idx])
			np.save("{}/labels{}.npy".format("data/eval", eval_nr), eval_labels[:eval_idx])
	print("DONE")


