/*
 * driveforward.cc
 *
 *  Created on: 14 jan. 2014
 *      Author: Amir Shantia
 */

#include "robotdriver.ih"

bool RobotDriver::driveForwardOdom(double distance, double moveSpeed)
{
//wait for the listener to get the first message

//we will record transforms here
	double tf_wait = 5.0;
	double command_wait = 10.0;
	double stop_time = 0.5;
	tf::StampedTransform start_transform;
	tf::StampedTransform current_transform;
	ros::Time tf_now;
	try
		{
		tf_now = ros::Time::now();
		listener_.waitForTransform("base_link", "odom", tf_now,
				ros::Duration(tf_wait));
		listener_.lookupTransform("base_link", "odom", tf_now,
				start_transform);
		} catch (tf::TransformException &ex)
		{
			ROS_ERROR("%s", ex.what());
			return false;
		}

//record the starting transform from the odometry to the base frame


//we will be sending commands of type "twist"
	geometry_msgs::Twist base_cmd;
//the command will be to go forward at 0.25 m/s
	base_cmd.linear.y = base_cmd.angular.z = 0;
	base_cmd.linear.x = moveSpeed;

	bool done = false;
	ros::Time now = ros::Time::now();
	while (!done && nh_.ok() && !stop)
	{
		if ((ros::Time::now() - now).toSec() > command_wait)
		{
			ROS_WARN("Time is up for Move.");
			ROS_DEBUG_STREAM("T1: " << ros::Time::now().toSec() << "T2: " << now.toSec() << "Dur: " << (ros::Time::now() - now).toSec() );
			break;
		}
		//send the drive command
		cmd_vel_pub_.publish(base_cmd);
		loop_rate.sleep();
		//get the current transform
		try
		{
			tf_now = ros::Time::now();
			listener_.waitForTransform("base_link", "odom", tf_now,
						ros::Duration(tf_wait));
			listener_.lookupTransform("base_link", "odom",
					tf_now, current_transform);
		} catch (tf::TransformException &ex)
		{
			ROS_ERROR("%s", ex.what());
			break;
		}
		//see how far we've traveled
		tf::Transform relative_transform = start_transform.inverse()
				* current_transform;
		double dist_moved = relative_transform.getOrigin().length();

		if (dist_moved > distance)
		{
			ROS_DEBUG("Moving Finished.");
			done = true;
			break;
		}
	}

	now = ros::Time::now();
	while ((ros::Time::now() - now).toSec() < stop_time)
	{
		loop_rate.sleep();
		base_cmd.linear.y = base_cmd.angular.z = base_cmd.linear.x = 0;
		cmd_vel_pub_.publish(base_cmd);
	}

	if (false and client_reader.exists())
	{
		//Reverts the stop command
		if (stop)
			done = true;
			stop = false;

		borg_pioneer::MemorySrv srv;
		srv.request.timestamp = ros::Time::now();
		srv.request.name = "navigation_command_report";
		char jsonmsg[255];
		sprintf(jsonmsg, "{\"state\": %d \"time\": %f}", done, ros::Time::now().toSec());
		srv.request.json = std::string(jsonmsg);
		client_writer.call(srv);

		ROS_INFO_STREAM("Move Finished, status " << (done ? "OK" : "Fail"));
	}
	return done;
}

