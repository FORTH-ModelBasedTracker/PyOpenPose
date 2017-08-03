## PyOpenPose

Python bindings for the awesome Openpose library. 

Openpose github page:
https://github.com/CMU-Perceptual-Computing-Lab/openpose


### Building

Clone and build openpose. I assume you will also build caffe inside openpose 3rdparty folder.
Set and environment variable named OPENPOSE_ROOT pointing to the openpose root folder.

Inside the root folder of PyOpenpose execute the following commands:

```bash
mkdir build
cd build
cmake ..
make
```

Set your PYTHONPATH to point to the folder containing PyOpenPose.so


### Testing

Check the scripts folder for python examples using PyOpenPose.