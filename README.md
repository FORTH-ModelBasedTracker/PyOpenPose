## PyOpenPose

Python bindings for the awesome Openpose library. 



Openpose [github page](https://github.com/CMU-Perceptual-Computing-Lab/openpose)


### Building

Clone and build openpose. I assume you will also build caffe inside openpose 3rdparty folder.
Set an environment variable named OPENPOSE_ROOT pointing to the openpose root folder.

__Note:__ Openpose lib is under heavy development and the API changes very often. 
Some API changes will break PyOpenPose. I try to upgrade as soon as possible but I am usually a few days behing. 
Openning an issue helps to speed-up the proccess.

Current PyOpenPose version is built with openpose commit: [e086e6f](https://github.com/CMU-Perceptual-Computing-Lab/openpose/commit/e086e6fc6a5f4650d5c3a100b8ced8363fd2d099)

_Note:_ PyOpenPose requires __opencv3.x__. You will have to build openpose with opencv3 as well.

Inside the root folder of PyOpenpose run cmake and build with:

```bash
mkdir build
cd build
cmake ..
make
```

Add the folder containing PyOpenPose.so to your PYTHONPATH.


### Testing

Check the scripts folder for python examples using PyOpenPose.
