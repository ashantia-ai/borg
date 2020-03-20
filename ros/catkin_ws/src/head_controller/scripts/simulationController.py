#!/usr/bin/env python
import roslib
roslib.load_manifest('head_controller')

import rospy
import actionlib
from std_msgs.msg import Float64
import trajectory_msgs.msg 
import control_msgs.msg  
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryAction, JointTrajectoryGoal, FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import time


class Joint:
    def __init__(self, motor_name):
        #arm_name should be b_arm or f_arm
        self.name = motor_name           
        self.jta = actionlib.SimpleActionClient('/alice/pan_tilt_trajectory_action_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        rospy.loginfo('Waiting for joint trajectory action')
        self.jta.wait_for_server()
        rospy.loginfo('Found joint trajectory action!')
        
    def move_joint(self, angles):
        goal = FollowJointTrajectoryGoal()                      
        char = self.name[0] #either 'f' or 'b'
        goal.trajectory.joint_names = ['head_yaw_joint', 'head_pitch_joint']
        point = JointTrajectoryPoint()
        point.positions = angles
	#point.velocities = [0.1, 0.1]
        point.time_from_start = rospy.Duration(2)                       
        goal.trajectory.points.append(point)
        self.jta.send_goal_and_wait(goal)
        
    def forward_nav(self):
        tilt_nav = 0.40
        pan_nav = 0.0
        self.move_joint([pan_nav, tilt_nav])
        
    def left_nav(self):
        tilt_nav = 0.40
        pan_nav = 1.57
        self.move_joint([pan_nav, tilt_nav])

    def right_nav(self):
        tilt_nav = 0.40
        pan_nav = -1.57
        self.move_joint([pan_nav, tilt_nav])

    def object_rec(self):
        tilt_nav = 0.2
        pan_nav = 0.0
        self.move_joint([pan_nav, tilt_nav])
            
def main():
    arm = Joint('f_arm')

    arm.forward_nav()
 #   arm.left_nav()
 #   arm.object_rec()
 #   arm.right_nav()


                        
if __name__ == '__main__':
    rospy.init_node('joint_position_tester')
    main()

