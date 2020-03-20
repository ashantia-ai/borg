# borg
The Code to run and evaluate two-stage deep visual navigation using CNNs and reinforcement learning.

Requirements:
a. ROS Kinetic
b. Gazebo
c. Tensorflow (Version 1.9.0)
d. Numpy
e. OpenCV

1- Evaluating the CNN:
a. Please correct the path to the latest model in the following files:
borg/ros/catkin_ws/src/gazebo_controller/scripts/cafe_room_cnn_live_sudo.py line 416
borg/ros/catkin_ws/src/gazebo_controller/scripts/kitchen_cnn_live.py line line 396
b. Build the ros/catkin_ws using catkin build
c. Copy the folders inside the gazebo models in the .gazebo folder in your home directory 
d. run the omni_cafe.sh in the sudo_gazebo package, inside the launch folder.
e. Run Rviz and load the given rviz setting.
f. Select new navigation points and watch the CNN outputs from the red Arrows

The robot and its Environment
We have four environments.
Maze
One  maze type is a randomly generated matrix of a certain size (square size in our case). The maze is generated using the Maze_generator.py file in the cacla_nav package. You have to set the path inside the file. Read the script comments for more info.
The other maze type is a manual one. This one should be use to extract approximated maze from the precise map files of the environment. This maze is generated using Shower.py . Yes I know the name is ridiculous. The file requires the pgm map of the environment. You have to specify the required size of the maze, the number of start, and goal locations, and the saving path. The result is an image showing the maze with goal and start points, and the numpy matrices representing each.
MAKE SURE YOU KEEP THESE FILES. ESPECIALLY THE MAP GENERATED MAZES. You have yo use it for all the RL experiments.
2D simulator (stage ros)
So we use slightly altered stage ros. As long as you use amirNav branch, you can just compile the stage_ros-lunar-devel package and be done with it.
The package stage_nav has move_base parameters set and ready for the the stageros
The package sudo_stage has the world folder which includes the map both for the small kitchen and the bigroom environment. So besically the maps in the g_world folder inside this package should be used. The scripts will fire the correct files, and these are used by the following python files in Cacla_nav folder:
g_stage_nav.py
g_stage_nav_multigoal.py
g_stage_nav_multigoal_cafe.py

3D simulator (gazebo ros)
Make sure you have the latest gazebo from kinetic! USE ROS KINETIC. Gazebo has a tendency of screwing up light reflections on version changes.
All the required gazebo models are in the base folder of the amirNav branch. It is called “gazebo_models” duh! Make sure you copy this to your .gazebo folder in your home directory.
The amirNav has all the necessary packages to run the gazebo. 
Make sure you have the correct neural network models as well.
Selecting and Preparing CNN
We have two CNNs, one for the Small Kitchen , and one for the Bigroom. The scripts that run them are in the gazebo_controller package. And the scripts names are self explanatory. One has cafe room (big room), and the other kitchen in the name.
When you run it, it will subscribe to the correct topic. Note that small kitchen uses pioneer and big room uses alice.
The scripts publish a pose array on “/estimated_pose” topic. You can visualize it. If the arrows are not precise then you probably have the wrong model. Discuss with me, and we shall find the correct one.
Small Kitchen
 For the small kitchen, the launch file is in sudo_gazebo/launch/omni_cheery_advanced.launch. Yes, we use omni directional movement to make our lives easier.
You need to launch navigation using the borg2d_nav package. Launch file is borg_local_navigation.launch. 
The BigRoom
For the bigroom, we also need to use no collision and omni directional movement. Therefore, from the alice_gazebo package, launch the alice_simulation.launch. The urdf file is already set to use omni directional movement.
Note: there is no borg/alice pointcloud in amirnav, therefore, I only use the laser scanner for obstacle detection.
In addition, you need to make sure that alice head controller is pointing to the current direction (TO BE ADDED.). This is crucial since the CNN trainings where done with a certain angle
Real Robot (Not used for experiments)
The program is divided into several sections.
1. The Robot and its environment
2. The Reinforcement Learning Method
3. The analysis tool



