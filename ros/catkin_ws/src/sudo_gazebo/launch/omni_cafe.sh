#!/bin/bash

clear

number=$1
echo "Running Kitchen World $1"
xterm -geometry 93x31+1800+0 -e roslaunch sudo_gazebo omni_cafe.launch &

sleep 25
xterm -geometry 93x31+1800+600 -e roslaunch sudo_gazebo omni_cafe_dependencies.launch &

sleep 2
