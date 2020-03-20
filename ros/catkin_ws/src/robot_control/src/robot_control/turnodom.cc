/*
 * turnodom.cc
 *
 *  Created on: 14 jan. 2014
 *      Author: Amir Shantia
 */
#include "robotdriver.ih"

bool RobotDriver::turnOdom(bool clockwise, double radians, double turnSpeed)
{
	while (radians < 0)
		radians += 2 * M_PI;
	while (radians > 2 * M_PI)
		radians -= 2 * M_PI;

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
	//the command will be to turn at 0.75 rad/s
	base_cmd.linear.x = base_cmd.linear.y = 0.0;
	base_cmd.angular.z = turnSpeed;
	if (clockwise)
		base_cmd.angular.z = -base_cmd.angular.z;

	//the axis we want to be rotating by
	tf::Vector3 desired_turn_axis(0, 0, 1);
	if (!clockwise)
		desired_turn_axis = -desired_turn_axis;

	bool done = false;
	ros::Time now = ros::Time::now();
	while (!done && nh_.ok() && !stop)
	{
		if ((ros::Time::now() - now).toSec() > command_wait)
		{
			ROS_WARN("Time is up for Turn.");
			ROS_DEBUG_STREAM("T1: " << ros::Time::now().toSec() << "T2: " << now.toSec() << "Dur: " << (ros::Time::now() - now).toSec() );
			done = false;
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
		tf::Transform relative_transform = start_transform.inverse()
				* current_transform;

		tf::Vector3 actual_turn_axis =
				relative_transform.getRotation().getAxis();
		double angle_turned = relative_transform.getRotation().getAngle();


		if (fabs(angle_turned) < 1.0e-2)
			continue;

		if (actual_turn_axis.dot(desired_turn_axis) < 0)
			angle_turned = 2 * M_PI - angle_turned;

		//ROS_DEBUG_STREAM("Angle Turned: " << angle_turned << "\n Desired Radians: " << radians << '\n');
		if (angle_turned > radians)
		{
			ROS_DEBUG("Turn Finished\n");
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

		ROS_INFO_STREAM("Turn Finished, status " << (done ? "OK" : "Fail"));
	}
	return done;
}

