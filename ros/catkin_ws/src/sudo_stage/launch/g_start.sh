#!/bin/bash

clear

number=$1
echo "Running Kitchen World $1"
xterm -geometry 93x31+1800+0 -e roslaunch sudo_stage g_stage.launch &

sleep 2
xterm -geometry 93x31+1800+600 -e roslaunch stage_nav g_stage_navigation.launch &

sleep 2
