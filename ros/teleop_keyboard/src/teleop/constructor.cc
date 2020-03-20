#include "teleop.ih"
#include <geometry_msgs/Twist.h>
#include <turtle_actionlib/Velocity.h>
#include <string>

using namespace std;

namespace BORG
{
    TeleOp::TeleOp(ros::NodeHandle &nh, string const &topic, bool turtle)
    :
        d_handler(nh),
        d_terminalstate(0),
        d_turtlebot(turtle)
    {
        if (d_turtlebot)
            d_cmd_vel_pub = d_handler.advertise<turtle_actionlib::Velocity>(topic, 1);
        else
            d_cmd_vel_pub = d_handler.advertise<geometry_msgs::Twist>(topic, 1);

        setupTerminal();
    }
}
