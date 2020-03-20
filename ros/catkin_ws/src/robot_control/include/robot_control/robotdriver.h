#ifndef __INCLUDED_ROBOTDRIVER_H_
#define __INCLUDED_ROBOTDRIVER_H_

#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <iosfwd>
#include <std_msgs/Int8.h>
#include <robot_control/action.h>
#include <actionlib/server/simple_action_server.h>
#include <robot_control/rlAction.h>

class RobotDriver
{
protected:
	//! The node handle we'll be using
	ros::NodeHandle nh_;
	//! We will be publishing to the "cmd_vel" topic to issue commands
	ros::Publisher cmd_vel_pub_;
	//! We will be listening to TF transforms as well
	tf::TransformListener listener_;
	//Service client for borg memory
	ros::ServiceClient client_reader;
	ros::ServiceClient client_writer;
	ros::ServiceServer service;

	actionlib::SimpleActionServer<robot_control::rlAction> as_;
	robot_control::rlFeedback feedback_;
	robot_control::rlResult result_;


	// This class can also listen to a topic for controls
	ros::Subscriber action_sub_;

	//Rate of this class
	ros::Rate loop_rate;

	//Holds the time for last command
	double last_command;

	//Variable that functions checks to see whether they should stop.
	bool stop;
public:
	//! ROS node initialization
	RobotDriver();

	RobotDriver(std::string topic);

	//! Drive forward a specified distance based on odometry information
	bool driveForwardOdom(double distance, double moveSpeed = 0.1);

	//!Turn a specified Radians based on odometry information
	bool turnOdom(bool clockwise, double radians, double turnSpeed = 0.3);

	bool jsonParser(std::string &type, bool &clockWise, double &amount);

	//bool readAction(robot_control::action::Request &req, robot_control::action::Response &res);
	void readAction(robot_control::rlGoalConstPtr const &goal);
	//Immediately stops the robot and forces other functions to stop as well.
	void stopRobot();
};

#endif
