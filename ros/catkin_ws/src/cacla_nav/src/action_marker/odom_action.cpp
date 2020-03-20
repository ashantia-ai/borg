/*
 * action.cpp
 *
 *  Created on: 27/03/2017
 *      Author: amir
 */

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <cacla_nav/positions.h>
#include <geometry_msgs/Twist.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_broadcaster.h>
#include <cmath>
#include <iostream>
#include <ios>

ros::Publisher marker_pub, noexp_marker_pub;
ros::Subscriber sub;
// Set our initial shape type to be an arrow
uint32_t shape = visualization_msgs::Marker::ARROW;

void visualizeCB(cacla_nav::positions to_visualize)
{

	geometry_msgs::PoseStamped current = to_visualize.current;
	geometry_msgs::PoseStamped goal = to_visualize.goal;

	visualization_msgs::Marker marker, orig_marker;
	// Set the frame ID and timestamp.  See the TF tutorials for information on these.
	marker.header.frame_id = "/odom";
	marker.header.stamp = ros::Time::now();

	// Set the namespace and id for this marker.  This serves to create a unique ID
	// Any marker sent with the same namespace and id will overwrite the old one
	marker.ns = "action_arrow";
	marker.id = 0;

	// Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
	marker.type = shape;

	// Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
	marker.action = visualization_msgs::Marker::ADD;

	// Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
	marker.pose.position.x = current.pose.position.x;
	marker.pose.position.y = current.pose.position.y;
	marker.pose.position.z = 0.5;

	// Set the color -- be sure to set alpha to something non-zero!
	marker.color.r = 0.0f;
	marker.color.g = 1.0f;
	marker.color.b = 0.0f;
	marker.color.a = 1.0;

	marker.lifetime = ros::Duration(5);

	tf::Quaternion q;
        double pitch_up = -3.14;
	q.setRPY(0,pitch_up,0);


	marker.pose.orientation.x = q.getX();
	marker.pose.orientation.y = q.getY();
	marker.pose.orientation.z = q.getZ();
	marker.pose.orientation.w = q.getW();

	// Set the scale of the marker -- 1x1x1 here means 1m on a side
	marker.scale.x = 1.0;
	marker.scale.y = 0.1;
	marker.scale.z = 0.02;

	/********The original action marker*********/
	orig_marker = marker;
	orig_marker.ns = "noexp_action_arrow";
	orig_marker.id = 1;
	// Set the color -- be sure to set alpha to something non-zero!
        
	orig_marker.pose.position.x = goal.pose.position.x;
	orig_marker.pose.position.y = goal.pose.position.y;
	orig_marker.color.r = 0.0f;
	orig_marker.color.g = 0.0f;
	orig_marker.color.b = 1.0f;
	orig_marker.color.a = 1.0;

	q.setRPY(0,pitch_up,0);

	orig_marker.pose.orientation.x = q.getX();
	orig_marker.pose.orientation.y = q.getY();
	orig_marker.pose.orientation.z = q.getZ();
	orig_marker.pose.orientation.w = q.getW();

	// Set the scale of the marker -- 1x1x1 here means 1m on a side
	orig_marker.scale.x = 1.0;

	// Publish the marker
	while (marker_pub.getNumSubscribers() < 1)
	{
	  if (!ros::ok())
	  {
		return;
	  }
	  ROS_WARN_ONCE("Please create a subscriber to the marker");
	  sleep(1);
	}
	marker_pub.publish(marker);
	noexp_marker_pub.publish(orig_marker);


}

int main( int argc, char** argv )
{
  ros::init(argc, argv, "action_marker_odom");
  ros::NodeHandle n;
  ros::Rate r(20);
  marker_pub = n.advertise<visualization_msgs::Marker>("action", 1);
  noexp_marker_pub = n.advertise<visualization_msgs::Marker>("noexp_action", 1);
  sub = n.subscribe("/cacla_odom_actions", 1 ,visualizeCB);

  while(true)
  {
	  r.sleep();
	  ros::spinOnce();
  }
}




