import experiment as ep
import gazebo as gz

import tf
import rospy

import time

import rospkg

from os.path import *

# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# get the file path for rospy_tutorials
path = rospack.get_path('alice_description')

#TODO: Make this configurable:
door_sdf_location = join(path,"sdf/door.sdf")
fixed_door_sdf_location = join(path,"sdf/fixed_door_nocontact.sdf")

if __name__ == "__main__":
    
    rospy.init_node('btool', anonymous=True)
    #Insert doors with random pose
    ep.insert_doors(['open','open','open'])
    transform = tf.TransformListener();

    first_check = False
    second_check = False
    time_buffer = 60.0
    first_check_time = 0.0
    
    sleep_rate = rospy.Rate(5)
    
    while not first_check:
        sleep_rate.sleep()
        transform.waitForTransform("/odom", "/base_footprint", rospy.Time(0), rospy.Duration(2))
        trans, rot = transform.lookupTransform("odom", "base_footprint", rospy.Time(0))
        
        for i in range(1,4):
            x, y, theta = ep.door_positions['door_%s' % i]['closed']
            distance = ((trans[0] - x)**2 + (trans[1] - y)**2)**0.5
            
            if distance < 1.5:
                #The robot is closer than 1 meter to the door location.
                print "Robot is getting close to a door"
                first_check = True
                gz.delete_model('door_%s' % i)
                gz.pause_physics()
                gz.insert_sdf(fixed_door_sdf_location, 'door_%d' % i, x , y, 0) #Move the door to closed position
                
                gz.unpause_physics()
                #Adds a timer check for second door_check
                first_check_time = time.time()
                break
                
    #Empty loop until timer is gone
    while time.time() - first_check_time < time_buffer:
        sleep_rate.sleep()
        
    while not second_check:
        sleep_rate.sleep()
        transform.waitForTransform("/odom", "/base_link", rospy.Time(0), rospy.Duration(2))
        trans, rot = transform.lookupTransform("odom", "base_link", rospy.Time(0))
        
        for i in range(3):
            x, y, theta = door_positions['door_%s' % i]['closed']
            distance = ((trans[0] - x)**2 + (trans[1] - y)**2)**0.5
            
            if distance < 1.0:
                #The robot is closer than 1 meter to the door location.
                print "Robot is getting close to a door"
                second_check = True
                #Adds a timer check for second door_check
                break
    

