## PyOpenPose

Python bindings for the awesome Openpose library. 

Openpose [github page](https://github.com/CMU-Perceptual-Computing-Lab/openpose)


### Building

Clone and build openpose. If you use cmake then ```make install``` will copy 
all necessary headers and libs to an _install_ forder that you specify (default is ```/usr/local```). 

Set an environment variable named OPENPOSE_ROOT pointing to the openpose _install_ folder.
For running the example scripts make sure OPENPOSE_ROOT contains a models folder with the openpose models.

__Note:__ Openpose lib is under heavy development and the API changes very often. 
Some API changes will break PyOpenPose. I try to upgrade as soon as possible but I am usually a few days behind. 
Openning an issue helps to speed-up the proccess. 
Current PyOpenPose version is built with [openpose commit d621465](https://github.com/CMU-Perceptual-Computing-Lab/openpose/commit/d6214653740d8100658a6e4d224f94d3c4c673e1)

__Note:__ PyOpenPose requires __opencv3.x__. You will have to build openpose with opencv3 as well.

Inside the root folder of PyOpenpose run cmake and build with:

```bash
mkdir build
cd build
cmake ..
make
```

Add the folder containing PyOpenPose.so to your PYTHONPATH.

### Building the library for python3 or python2

 - Open CMakeLists.txt and uncomment the corresponding two lines for python discivery
 - rebuild project

### Testing

Check the scripts folder for python examples using PyOpenPose.
