# FAST-LIO-STD

## Introduction

This repository combines FAST-LIO with STD loop closure detection to achieve robust LiDAR SLAM with loop closure. It includes:

- Real-time LiDAR-inertial odometry based on FAST-LIO
- Loop closure detection using STD (Stable Triangle Descriptor)
- Loop closure optimization using GTSAM
- Mapping with pose graph optimization


### Output Files

The trajectory data (tum format) will be saved in the `Log` folder:
- Odometry trajectory path `lio.txt`
- Optimized trajectory path with loop closure detection `loop.txt`

The point cloud data (PCD format) will be saved in the `PCD` folder:
- Raw accumulated point cloud: `scans.pcd`
- Globally optimized point cloud map: `opt_map.pcd`

## Prerequisites

- ROS (tested on Noetic)
- PCL >= 1.10
- Eigen >= 3.3.4
- GTSAM >= 4.0
- OpenCV >= 4.2
- Ceres == 1.14
- livox_ros_driver

### Installation of Dependencies

```bash
# Install PCL
sudo apt install libpcl-dev
# Install Eigen3
sudo apt install libeigen3-dev
# Install GTSAM
sudo apt install ros-noetic-gtsam
# Install OpenCV
sudo apt install libopencv-dev
# Install Ceres
sudo apt install libceres-dev
```

## Build

```bash
cd ~/catkin_ws/src
git clone https://github.com/shitongbeep/FAST-LIO-STD.git
git clone https://github.com/Livox-SDK/livox_ros_driver
cd ..
catkin_make
```

## Run

```bash
source devel/setup.bash
roslaunch fast_lio_std mapping_xx.launch
rosbag play YOUR_BAG.bag
```

## Related Works

**SLAM:**

1. [ikd-Tree](https://github.com/hku-mars/ikd-Tree): A state-of-art dynamic KD-Tree for 3D kNN search.
2. [FAST-LIO](https://github.com/hku-mars/FAST_LIO): Fast Direct LiDAR-inertial Odometry.
3. [STD](https://github.com/hku-mars/STD): A Stable Triangle Descriptor for 3D place recognition.
