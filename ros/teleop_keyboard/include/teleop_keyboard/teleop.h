#ifndef __INCLUDED_ABOT_TELEOP_H_
#define __INCLUDED_ABOT_TELEOP_H_

#include <ros/ros.h>
#include <string>
#include <termios.h>

namespace BORG
{
    class TeleOp
    {
        private:
            ros::NodeHandle d_handler;
            ros::Publisher d_cmd_vel_pub;
            termios d_start_tcflags;
            int d_start_fcflags;
            int d_terminalstate;
            bool d_turtlebot;

        public:
            TeleOp(ros::NodeHandle &nh, std::string const &topic, bool turtle = false);
            ~TeleOp();

            bool run();

        private:
            bool setupTerminal();
            bool restoreTerminal();
    };
}
#endif
