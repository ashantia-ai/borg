from geometry_msgs.msg import Point
import rospy, roslib


if __name__ == '__main__':
    
    rospy.init_node('test')

    publisher = rospy.Publisher('/some_force', Point)
    data = Point()
    data.x = 3.66
    data.y = 1.0
    data.z = 0
    for i in range(100000):
        publisher.publish(data)
    print "done"
