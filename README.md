# VINS-DIO: VIO with Deep Inertial Velocity Prediction

## 1. build with Torch
first install cuda 11.3 (same version with cudatoolkit in my conda env.) and cudnn at /usr/local/cuda-*, then download libtorch and unzip
notice: be sure to use cxx11-abi version, for example, url: https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.12.0%2Bcu113.zip

### 2. why libtorch's speed is low
1. GPU freq.: no
query GPU freq.: sudo nvidia-smi -q -d SUPPORTED_CLOCKS
fix into max GPU freq.: sudo nvidia-smi -lgc 7501
set the GPU freq. free again: sudo nvidia-smi -rgc
2. memory: to do
3. Nvidia driver version (400~500 is stable): to do


source /home/oran/WS/Work/SLAM/VINS-DIO/devel/setup.bash
roslaunch vins_estimator tum_odometry_no_rviz.launch

rosbag play /home/oran/WS/WSb/DataSet/TUM_VI/rosbag/dataset-magistrale4_512_16.bag