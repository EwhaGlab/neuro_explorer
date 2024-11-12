# Neuro-Explorer
**Neuro-Explorer** package is a 2D mobile robot's exploration system based on the learned frontier region detectors.
This package contains C/C++ deployer of the trained models discussed in our IROS24 paper: [Neuro-Explorer: Efficient and Scalable Exploration Planning via Learned Frontier Regions](http://graphics.ewha.ac.kr/neuro_explorer/)
Please refer to the paper if you have inquiries regarding the technical aspect of this package.

## What's inside

**Neuro-Explorer** consists of two seperable projects: **neuro_explorer** and **neuro_explorer_train**. The former contains the exploration planner which employs pre-trained artifical neural network models, while the latter includes Python scripts for training the network models. Although **neuro_explorer_train** is nested within **neuro_explorer** for easier maintenance, the two projects operates independently.

## Dependencies

**neuro_explorer** is a ROS-compatible package. Therefore, we expect you to have Ubuntu 20.04 installed along with ROS Noetic.
The package has not been tested with ROS2 yet, but we will update this repository once it is compatible with ROS2.

This package requires ROS navigation stack to control an embodied agent. 
We recommend you to install our customized version of [navigation stack](https://github.com/han-kyung-min/navigation).

If you want to run this package in a synthetic environment, such as the Gazebo simulator, we recommend you install a mapping SW such as OctoMap. Use [Turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3) packages to explore your favorite world. 
Besides, we found [TEB](https://github.com/rst-tu-dortmund/teb_local_planner) local planner runs nicely with this package, so we recommend you to install this local planner.
In the case of solving real-world exploration problems with a mobile robot, you will need a SLAM SW to produce a 2D occupancy grid map. We recommend installing [SLAM toolbox](https://github.com/SteveMacenski/slam_toolbox) for your localization and mapping.

**neuro_explorer** deploys pre-trained network models trained in a [TensorFlow](https://www.tensorflow.org/install?hl=ko) environment. Refer to [neuro_explorer_train](./neuro_explorer_train/README.md) to see how to install tensorflow and its dependencies for re-training the network models.
To integrate the trained models into the C++ project, we utilize the [TensorFlow C API library (v2.12)](https://www.tensorflow.org/install/lang_c). Ensure this package is installed before installing our package.

## To install

### 1. Follow the steps below to install neuro_explorer_train and its dependencies
> #### (1) Clone this repo 

```
  cd ~/catkin_ws/src
  git clone https://github.com/EwhaGlab/neuro_explorer.git
```
> #### (2) Install [neuro_explorer_train](./neuro_explorer_train/README.md) 

> #### (3) Install [TensorFlow C API library (v2.12)](https://www.tensorflow.org/install/lang_c) based on its instruction

### 2. Install the required ROS navigation packages

> #### (1) Install Octomap 


```
  cd ~/catkin_ws/src
  git clone https://github.com/han-kyung-min/octomap_mapping.git
  cd octomap_mapping
  git checkout explore_bench-nn_burger-fast_gridmap_pub
  sudo apt-get install ros-<ros_ver>-octomap*
  sudo apt-get install ros-<ros_ver>-octomap-server
  cd ~/catkin_ws
  catkin_make install
```

> #### (2) Install Turtlebot3 package
```
  sudo apt-get install ros-<ros_ver>-turtlebot3

```
> #### (3) Install navigation stack
```
  cd ~/catkin_ws/src
  git clone https://github.com/han-kyung-min/navigation.git
  cd navigation
  git checkout proximity_check
  cd ~/catkin_ws
  catkin_make install
```

> #### (4) Install teb_local_planner
```
  cd ~/catkin_ws/src
  git clone https://github.com/rst-tu-dortmund/teb_local_planner
  cd teb_local_planner
  git checkout <your_ros_version_branch>
  cd ~/catkin_ws
  catkin_make -DCATKIN_WHITELIST_PACKAGES="teb_local_planner"
```

### 3. Install neuro_explorer

> #### (1) Modifying the config file
>> Open `<base_dir>/params/neuroexploration.yaml`, then modify the following fields according to your requirements: `debug_data_save_path:`, `fd_model_filepath:`, `astar_model_filepath:`, and `covrew_model_filepath:`.

>> where <base_dir> refers to `~/catkin_ws/src/neuro_explorer`


> #### (2) Install the package
```
  cd ~/catkin_ws
  catkin_make install
```

### 4. Download pre-trained models

> The pre-trained models are available at [**THIS LINK**](https://drive.google.com/drive/folders/1mXkKHI6-BrAemQjoGyCWZyQMnNZOVVh9?usp=sharing). Download and place them in `~/catkin_ws/src/neuro_explorer/nn_models/`. While the pre-trained models are sufficient to explore a large space such as WGx3, you have the option to retrain the model. For the retraining procedure, refer to [neuro_explorer_train](./neuro_explorer_train/README.md).


## Exploring WGx3 world  

> 1. Open up three terminals followed by the command below on the each terminal window.
> 2. In each terminal, run the command: `source ~/catkin_ws/install/setup.bash`. 
> 3. Run the commands below in each terminal window. 
```
  conda activate neuro_ae; roslaunch neuro_explorer WGx3.launch
  roslaunch neuro_explorer explorer_bench.launch
  roslaunch neuro_explorer neuro_explorer.launch
```

## Citaion
Good luck with your projects! Please cite our paper if you think **Neuro-Explorer** is helpful for your research work.

```
K.M. Han and Y.J. Kim, "Neuro-Explorer: Efficient and Scalable Exploration Planning via Learned Frontier Regions," 
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024
```

#### Feel free to send an email to *hankm@ewha.ac.kr* if you are having a trouble with compiling this package.
