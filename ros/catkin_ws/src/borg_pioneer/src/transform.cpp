#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

// This function is called whenever new data arrives at the odom topic
void poseCallback(const nav_msgs::Odometry::ConstPtr& odomsg)
{
    //TF odom => base_link
    static tf::TransformBroadcaster odom_broadcaster;
    odom_broadcaster.sendTransform(tf::StampedTransform(
        tf::Transform(tf::Quaternion(
            odomsg->pose.pose.orientation.x,
            odomsg->pose.pose.orientation.y,
            odomsg->pose.pose.orientation.z,
            odomsg->pose.pose.orientation.w
        ), tf::Vector3(
            odomsg->pose.pose.position.x, 
            odomsg->pose.pose.position.y, 
            odomsg->pose.pose.position.z
        )), odomsg->header.stamp, "/odom", "/base_link"
	));
    ROS_DEBUG("odometry frame sent");
}

int main(int argc, char** argv) 
{
    // ROS initialization stuff
	ros::init(argc, argv, "pioneer_tf");
	ros::NodeHandle n;
	ros::Rate r(20);
    
    tf::TransformBroadcaster broadcaster;
  
    // Subscribe to pose info
	ros::Subscriber pose_sub = n.subscribe<nav_msgs::Odometry>("odom", 1, poseCallback);
    while(n.ok())
    {
        //base_link => laser
		broadcaster.sendTransform(
			tf::StampedTransform(
				//For laser rotate on Z Axis 180 degrees.
				tf::Transform(tf::Quaternion(0, 0, M_PI), tf::Vector3(-0.15, 0.0, 0.2)),
				ros::Time::now(), "/base_link", "/laser"));
        ros::spinOnce();		
        r.sleep();
	}
}				

