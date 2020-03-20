'''
It prints grid location based on type of the given map
'''

import rospy
from nav_msgs.msg import Odometry
import tf

def process_odometry(self, data):
    dict = {}
    dict.add(data.twist.twist.linear.x, 'lin_speed')
    dict.add(data.twist.twist.linear.z, 'ang_speed')
    dict.add(data.pose.pose.position.x, 'x')
    dict.add(data.pose.pose.position.y,'y')
    dict.addrotation = (data.pose.pose.orientation.x, data.pose.pose.orientation.y,
               data.pose.pose.orientation.z, data.pose.pose.orientation.w) 
    
    degrees = tf.transformations.euler_from_quaternion(rotation)
    degrees = math.degrees(degrees[2])
    sin_theta = math.sin(degrees[2])
    cos_theta = math.cos(degrees[2])
    radian = degrees[2]
    
def convert_to_grid(x, y, theta):
    '''
    Converts the location output to grid like format
    '''
    
    columns = x_len #int(math.ceil(self.x_len / self.metric_resolution))
    rows = y_len #int(math.ceil(self.y_len / self.metric_resolution))
            
    x_idx = (y) / -step_size_x
    x_idx = int(x_idx)
    y_idx = (x) / step_size_y
    y_idx = int(y_idx)
    
    
    return x_idx, y_idx
          
def get_yaw(data):
    rotation = (data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                   data.pose.pose.orientation.z, data.pose.pose.orientation.w) 
        
    degrees = tf.transformations.euler_from_quaternion(rotation)
    return degrees[2]

if __name__== '__main__':
    step_size_y = step_size_x = 0.4
    x_len = 47
    y_len = 24
    rospy.init_node("gridd", anonymous=True)
    
    while True:
        try:
            data = rospy.wait_for_message('/odom', Odometry)
            print "Current Grid Cell", convert_to_grid(data.pose.pose.position.x, data.pose.pose.position.y, get_yaw(data))
        except KeyboardInterrupt:
            "Exit."
            break