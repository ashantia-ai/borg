#include "robotdriver.ih"

RobotDriver::RobotDriver(std::string topic)
:
last_command(0),
loop_rate(10),
stop(false),
as_(nh_, "readaction", boost::bind(&RobotDriver::readAction, this, _1), false)
{
    as_.start();
    //set up the publisher for the cmd_vel topic
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("cmd_vel", 1);


    //action_sub_ = nh_.subscribe<std_msgs::Int8>(topic, 1, &RobotDriver::readAction, this);
    //service = nh_.advertiseService("perform_action", &RobotDriver::readAction, this);
    ROS_INFO("Ready to perform actions");
};
