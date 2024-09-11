# Neuro-Explorer
**Neuro-Explorer** package is a 2D mobile robot's exploration system based on the learned frontier region detectors.
This package contains C/C++ deployer the trained models elaborated in our IROS24 paper: [Neuro-Explorer: Efficient and Scalable Exploration Planning via Learned Frontier Regions](http://graphics.ewha.ac.kr/neuro_explorer/)
Please refer to the paper if you have inquiries regarding the technical aspect of this package.

## Dependencies

**Neuro-Explorer** is a ROS-compatible package. Therefore, we expect you to have Ubuntu 20.04 installed along with ROS Noetic.
We have not tested the package with ROS2 yet, but we will update this repo as soon as the package is ready for ROS2 also.

You need the ROS navigation stack to control an embodied agent. 
Neuro-Explorer runs best with our customized version of [navigation stack](https://github.com/han-kyung-min/navigation).  

If you want to run this package in a synthetic environment, such as the Gazebo simulator, we recommend you install a mapping SW such as
OctoMap. Use [Turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3) packages to explore your favorite world. 
Besides, we found [TEB](https://github.com/rst-tu-dortmund/teb_local_planner) local planner runs OK with this package, so we recommend you to install this local planner.
In the case of solving real-world exploration problems with a mobile robot, you will need a SLAM SW to produce a 2D occupancy grid map. 
We recommend installing [SLAM toolbox](https://github.com/SteveMacenski/slam_toolbox) for your localization and mapping.

**Neuro-Explorer** deploys pre-trained network models in a [TensorFlow](https://www.tensorflow.org/install?hl=ko) environment. Refer to [neuro_ae]() to see how to install tensorflow and its dependencies. 
To deploy the trained models in the C++ project, we utilize [TensorFlow C API library](https://www.tensorflow.org/install/lang_c). Make sure to install this package prior to install our package.


## To install base libraries and the conda environment

Follow [TensorFlow C API library](https://www.tensorflow.org/install/lang_c)'s instruction

Follow [neuro_ae]()'s instruction 

## Example 1: To run Neuro-Explorer (NE) in WGx3 world

If you are running NE on a synthetic environment such as a Gazebo world, follow the instructions below.


### Clone TEB local planner

```
cd ~/catkin_ws/src
git clone https://github.com/rst-tu-dortmund/teb_local_planner.git
cd ~/catkin_ws
catkin_make install

```
### Install Turtlebot3 package
```
sudo apt-get install ros-<ros_ver>-turtlebot3
```
### Clone OctoMap (our customized version)
```
cd ~/catkin_ws/src
git clone https://github.com/han-kyung-min/octomap_mapping.git
git checkout explore_bench-nn_burger-fast_gridmap_pub
```
### Clone NavStack (our customized version)
```
cd ~/catkin_ws/src
git clone https://github.com/han-kyung-min/navigation.git
git checkout proximity_check
```

## Install Neuro-Explorer 
```
cd ~/catkin_ws/src
git clone https://github.com/EwhaGlab/neuro_explorer.git
cd ~/catkin_ws
catkin_make install
```

### Start the exploration task
Open up three terminals followed by the command below on the each terminal window
Don't forget to "source ~/catkin_ws/install/setup.bash" before starting the launch files below
```
conda activate neuro_ae; roslaunch neuro_explorer WGx3.launch
roslaunch neuro_explorer explorer_bench.launch
roslaunch neuro_explorer neuro_explorer.launch
```

## Citation
Good luck with your projects! Please cite our paper if you think **Neuro-Explorer** is helpful for your research work.

```
K.M. Han and Y.J. Kim, "Neuro-Explorer: Efficient and Scalable Exploration Planning via Learned Frontier Regions," 
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024
```

Feel free to send us an email if you are having a trouble with compiling this package.
