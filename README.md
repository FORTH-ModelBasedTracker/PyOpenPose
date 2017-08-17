## PyOpenPose

Python bindings for the awesome Openpose library. 

Openpose github page:
https://github.com/CMU-Perceptual-Computing-Lab/openpose

notice!! PyOpenPose use OpenCV 3.2 and protobuf 2.6.1


### Install OpenCV 3.2 in ubuntu 16.04 or 14.04
you can install dependencies of Opencv using the follow command.
```
sudo apt-get install -y build-essential cmake pkg-config
sudo apt-get install -y libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y python2.7-dev python3.5-dev
```
download the source code
```
cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/3.2.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.2.0.zip
unzip opencv.zip
unzop opencv_contrib.zip
```

compile opencv without dnn(if you compile opencv with dnn, there will be an error.)
```
cd opencv/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D BUILD_opencv_dnn=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D PYTHON_EXECUTABLE=/usr/bin/python2.7 \
    -D BUILD_EXAMPLES=ON ..
```
install opencv
```
sudo make install
sudo ldconfig
```
you can also uninstall opencv using `sudo make uninstall`

### Installation of protobuf in ubuntu 16.04 or 14.04

```
sudo apt install libprotobuf-dev protobuf-compiler libprotobuf-lite9v5 protobuf9v5 libprotobuf-c1
```


### Installation of openpose 
please refer to openpose github 

first you have to compile the internal caffe in openpose
and then `make distribute`

second, you have to compile the openpose 
and then `make distribute`

PyOpenPose will use distribute.

### Installation of PyOpenPose
download the source of PyOpenPose
```
git clone https://github.com/FORTH-ModelBasedTracker/PyOpenPose.git
cd PyOpenPose
```

set the environment of openpose
```
export OPENPOSE_ROOT=/path/to/openpose
```

compile PyOpenPose
```
mkdir build
cd build 
# you have to specify the version of python or cmake will find the wrong version of python
cmake ..  -DPYTHON_INCLUDE_DIR=/usr/include/python2.7 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7.so

make all -j $nproc
# test PyOpenPose
cd ..
cd scripts
python OpLoop.py
```
Add the folder containing PyOpenPose.so to your PYTHONPATH, you can use PyOpenPose in you project now.
### Usage 
Check the scripts folder for python examples using PyOpenPose.

### Known Issues
*OpenCV 3.3 or 2.4 can not work with PyOpenPose

*Protobuf 3.3 cannot work with PyOpenPose
