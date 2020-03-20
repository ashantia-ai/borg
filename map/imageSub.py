#!/usr/bin/python
import threading
import signal  ##for handling OS signals (e.g. ctrl+c)
import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy

class ImageCallback():
	def __init__(self, topicname, callback):
		self.topic = topicname
		self.bridge = CvBridge()
		self.outerCB  = callback
		self.exit = False
		self.lock = threading.RLock()
		self.exitLock = threading.RLock()
		
		self.thread = threading.Thread(target = self.process)
		signal.signal(signal.SIGINT, self.signalCB)
		
		self.hasFrame = False
		self.image_sub = rospy.Subscriber(topicname,Image,self.generalCB,queue_size=1)
	
	def signalCB(self,signum, frame):## Handle ctrl + c manually. We need to clean-up multithreading stuff before exitting
		print "clean memory"
		self.target = None

		self.exitLock.acquire()
		self.exit = True
		self.exitLock.release()
		self.thread.join()
		sys.exit(0)
		
	def generalCB(self, data): ##callback function that sets the most recent data to process
		#print "genCB"
		self.lock.acquire()
		self.target = data
		self.lock.release()
		if self.hasFrame == False:
			self.thread.start()
		self.hasFrame = True
		
	def process(self):  ##executed by seperate thread to do the actual processing
		print "thread started"
		self.lock.acquire()
		while self.target != None:
			

			#print "process iter"
			data = self.target
			self.lock.release()

			self.exitLock.acquire() 
			print self.exit
			if self.exit == True:
				self.exitLock.release()
				break
			self.exitLock.release()
			try:
				cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  ##transform incomming data to an openCV image.
				#do stuff with image
				print "got an image"
				self.outerCB(cv_image)

				
			except CvBridgeError as e:
				print(e)
			
			self.lock.acquire()
		if self.target == None:
			self.lock.release()
			
def cb(image):
	cv2.imshow("image", image)
	cv2.waitKey(1)
if __name__ == '__main__':
	rospy.init_node('test', anonymous=True)
	test = ImageCallback("/front_xtion/rgb/image_raw", cb)
	rospy.spin()
