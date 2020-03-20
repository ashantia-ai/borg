/*
 * robotcontrol.cc
 *
 *  Created on: 14 jan. 2014
 *      Author: Amir Shantia
 */
#include <robot_control/robotdriver.h>
#include <borg_pioneer/MemoryReadSrv.h>
#include <ros/console.h>

#include <string>
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
	//init the ROS node
	ros::init(argc, argv, "robot_driver");
	RobotDriver driver;

	while (ros::ok())
	{
		string type;
		bool clockWise;
		double amount;
		if (driver.jsonParser(type, clockWise, amount))
		{
			double movespeed = 0.3;
			if (type == "move")
			{
				ROS_INFO_STREAM("Move Command. Amount(m): " << amount);
				driver.driveForwardOdom(amount, movespeed);
			}
			else if (type == "turn")
			{
				double turnspeed = 0.3;
				if (amount < 0.5)
					turnspeed = 0.1;
				ROS_INFO_STREAM(
						(clockWise ? "C" : "C-C") << " Turn. Amount(R): " << amount);
				driver.turnOdom(clockWise, amount, turnspeed);
			}
			else
			{
				driver.stopRobot();
			}
		}
	}
}


