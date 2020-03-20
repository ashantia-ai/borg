#include "teleop.ih"
#include <geometry_msgs/Twist.h>
#include <turtle_actionlib/Velocity.h>
#include <iostream>

using namespace std;

namespace BORG
{
    bool TeleOp::run()
    {
        if (!d_cmd_vel_pub)
        {
            ROS_ERROR("ROS Topic was not initialized properly. Make sure "
                      "it is of the correct type.");
            return false;
        }

        cout << "Control the robot by pressing WASD\n"
                "Press 'q' to exit\n"
                "Press ' ' to stop the robot\n"
                "When capslock is on, the speed will not decrease automatically\n";

        geometry_msgs::Twist base_cmd;
        turtle_actionlib::Velocity t_base_cmd;

        double speed = 0.0;
        double angle = 0.0;
        double const speed_step = 0.02;
        double const angle_step = .5;
        double const max_angle = 10.0;
        double const max_speed = 20.0;

        bool speed_locked = false;
        ros::Rate ticker(10);
        bool quit = false;

        while (d_handler.ok() and not quit)
        {
            int cmd = getchar();

            // Empty buffer
            while (getchar() != -1)
                ;

            base_cmd.linear.x = base_cmd.linear.y = base_cmd.angular.z = 0;   
            switch (cmd)
            {
                case 'w':
                    speed_locked = false;
                    ROS_DEBUG("Increasing speed");
                    speed += speed_step;
                    break;
                case 'W':
                    speed_locked = true;
                    ROS_DEBUG("Increasing speed");
                    if (speed >= speed_step)
                        speed *= 1.1;
                    else if (speed <= -speed_step)
                        speed = 0;
                    else
                        speed = speed_step;
                    break;
                case 's':
                    speed_locked = false;
                    ROS_DEBUG("Decreasing speed");
                    speed -= speed_step;
                    break;
                case 'S':
                    speed_locked = true;
                    ROS_DEBUG("Decreasing speed");
                    if (speed <= -speed_step)
                        speed *= 1.1;
                    else if (speed >= speed_step)
                        speed = 0;
                    else
                        speed = -speed_step;
                    break;
                case 'a':
                case 'A':
                    ROS_DEBUG("Turning left");
                    angle += angle_step;
                    break;
                case 'd':
                case 'D':
                    ROS_DEBUG("Turning right");
                    angle -= angle_step;
                    break;
                case ' ':
                    ROS_INFO("Stopping");
                    speed = 0;
                    angle = 0;
                    break;
                case 'q':
                case 'Q':
                    speed = angle = 0;
                    quit = true;
                    ROS_INFO("Quitting");
                default:
                    ;
            }

            if (speed > max_speed)
                speed = max_speed;
            else if (speed < -max_speed)
                speed = -max_speed;
            else if (angle > max_angle)
                angle = max_angle;
            else if (angle < -max_angle)
                angle = -max_angle;

            if (d_turtlebot)
            {
                t_base_cmd.linear = speed * 10.0;
                t_base_cmd.angular = angle;

                d_cmd_vel_pub.publish(t_base_cmd);
            }
            else
            {
                base_cmd.linear.x = speed * 10.0;
                base_cmd.angular.z = angle;

                d_cmd_vel_pub.publish(base_cmd);
            }


            // Decay the speed and angle to 0
            if (not speed_locked)
                speed *= 0.9;
            angle *= 0.75;

            // Make the 0 when they are really close
            if (abs(speed) < 0.001)
                speed = 0.0;
            if (abs(angle) < 0.001)
                angle = 0.0;
     
            ticker.sleep();
        }
        return true;
    }
}
