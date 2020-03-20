#include <iostream>
#include <teleop_keyboard/teleop.h>
#include <ros/ros.h>

using namespace std;
using namespace BORG;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "teleop_keyboard"); 
    ros::NodeHandle nh; 

    string topic = "/cmd_vel";
    bool turtlebot = false;

    if (argc > 1)
        topic = argv[1];

    if (topic.find("turtle") != string::npos)
        turtlebot = true;

    ROS_INFO("Publishing commands to topic %s", topic.c_str());
    if (turtlebot)
        ROS_INFO("Topic message type: turtlesim/Velocity");
    else
        ROS_INFO("Topic message type: geometry_msgs/Twist");

    TeleOp teleop(nh, topic, turtlebot);
    teleop.run();
}
