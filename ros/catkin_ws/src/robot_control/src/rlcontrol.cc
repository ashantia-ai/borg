/*
 * rlcontrol.cc
 *
 *  Created on: Jan 7, 2016
 *      Author: amir
 */
#include <robot_control/robotdriver.h>
#include <ros/console.h>

#include <string>
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
	//init the ROS node
	ros::init(argc, argv, "rl_robot_driver");

	RobotDriver driver("rl_actions");

	ros::spin();
}




