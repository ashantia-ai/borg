/*
 * driveforward.cc
 *
 *  Created on: 14 jan. 2014
 *      Author: Amir Shantia
 */

#include "robotdriver.ih"

enum rotation_type
{
	clockwise = true, cclockwise = false
};
void RobotDriver::readAction(robot_control::rlGoalConstPtr const &goal)
{
	ROS_INFO("ACTION CALLED");
	bool result = false;
	int action_int = goal->action;

	if (as_.isPreemptRequested() || !ros::ok())
	{
		ROS_INFO("readaction: Preempted");
		// set the action state to preempted

		result_.result = false;
		as_.setPreempted(result_, "pre-empted");
		return;
	}
	//double default_distance = 0.25; //Meters
	//double default_rotation = 0.785; //Radian
	double default_distance = 0.5; //Meters
	double default_rotation = 1.57; //Radian
	double default_x = 0.4; //M/s
	double default_theta = 1.0; //Radian/s

	switch (action_int)
	{
	case 0:
		result = driveForwardOdom(default_distance, default_x);
		break;
	case 1:
		result = driveForwardOdom(default_distance, -default_x);
		break;
	case 2:
		result = turnOdom(clockwise, default_rotation, default_theta);
		break;
	case 3:
		result = turnOdom(cclockwise, default_rotation, default_theta);
		break;
	case 4:
		stopRobot();
		result = true;
		break;
	}

	result_.result = result ? 0 : 1;
	if (result_.result == 0)
		as_.setSucceeded(result_, "done");
	else
		as_.setAborted(result_, "failed");

}

