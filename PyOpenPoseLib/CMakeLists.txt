set(SOURCE_FILES OpenPoseWrapper.cpp OpenPoseWrapper.h PyOpenPose.cpp opencv_expose.h opencv_expose.cpp)
add_library(PyOpenPose SHARED ${SOURCE_FILES})

target_link_libraries(PyOpenPose ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} openpose caffe)

pythonize(PyOpenPose)

