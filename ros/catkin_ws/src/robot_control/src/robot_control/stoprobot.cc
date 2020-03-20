/*
 * stoprobot.cc
 *
 *  Created on: Feb 27, 2014
 *      Author: Amir
 */


#include "robotdriver.ih"

void RobotDriver::stopRobot()
{
	//stop = true;
	geometry_msgs::Twist base_cmd;
	//the command will be to go forward at 0.25 m/s


	ros::Time now = ros::Time::now();
	while ((ros::Time::now() - now).toSec() < 0.5)
	{
		loop_rate.sleep();
		base_cmd.linear.y = base_cmd.angular.z = base_cmd.linear.x = 0;
		cmd_vel_pub_.publish(base_cmd);
	}
	ROS_DEBUG("Stopping Done.");
}
