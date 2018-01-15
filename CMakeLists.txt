cmake_minimum_required(VERSION 3.5)
project(PyOpenPose)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 11)
include(FeatureSummary)

##### Auto configure install prefix and build mode #####
if(DEFINED ENV{PYOPENPOSE_ROOT})
    message("PYOPENPOSE_ROOT is defined. Setting CMAKE_INSTALL_PREFIX to: $ENV{PYOPENPOSE_ROOT}")
    set(CMAKE_INSTALL_PREFIX $ENV{PYOPENPOSE_ROOT} CACHE PATH "Install path for PyOpenpose" FORCE )
endif()
message("Install prefix is set to: ${CMAKE_INSTALL_PREFIX}")

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
message("Generating with build type: ${CMAKE_BUILD_TYPE}")

###############

# Use at every target to make it default installable
macro(make_installable project)
    set_target_properties(${project} PROPERTIES DEBUG_POSTFIX "D")
    install(TARGETS ${project} EXPORT ${project}
		INCLUDES DESTINATION include
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
	install(EXPORT ${project} DESTINATION cmake)
endmacro()

# Use for every python target to take care of peculiarities
macro(pythonize project)
	set_target_properties(${project} PROPERTIES PREFIX "")
	if(WIN32)
            set_target_properties(${project} PROPERTIES SUFFIX ".pyd" FOLDER "Python")
	endif()
    make_installable(${project})
endmacro()

if(NOT WIN32) # compile with fPIC when building static with gcc
    set(CUDA_NVCC_FLAGS "--compiler-options=-fPIC ${CUDA_NVCC_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-fPIC ${CMAKE_C_FLAGS}")
endif()


find_package(CUDA REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(Doxygen)

### for python 2 use this:
find_package(Boost COMPONENTS system python-py27 REQUIRED)
find_package(PythonLibs 2.7 EXACT REQUIRED) 

### for python 3 use this:
#find_package(Boost COMPONENTS system python-py35 REQUIRED)
#find_package(PythonLibs 3 EXACT REQUIRED) 




# USE_CAFFE define needed by openpose headers
if(DEFINED ENV{OPENPOSE_ROOT})
    add_definitions("-DUSE_CAFFE")

    set(OPENPOSE_INCLUDE_DIRS "$ENV{OPENPOSE_ROOT}/include")
    set(OPENPOSE_LIBRARIES "$ENV{OPENPOSE_ROOT}/lib")
    set(CAFFE_INCLUDE_DIRS "$ENV{OPENPOSE_ROOT}/include")
    set(CAFFE_LIBRARIES "$ENV{OPENPOSE_ROOT}/lib")

else()
    message(FATAL_ERROR "Environment variable OPENPOSE_ROOT is not set. Please set and point to openpose installation folder")
endif()


link_directories(${CAFFE_LIBRARIES} ${OPENPOSE_LIBRARIES})
include_directories(${OPENPOSE_INCLUDE_DIRS})
include_directories(${CAFFE_INCLUDE_DIRS})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

feature_summary(WHAT ALL)


add_subdirectory(PyOpenPoseLib)
if(DOXYGEN_FOUND)
    add_subdirectory(doc)
endif()

# custom install target, needed for Clion IDE.
add_custom_target(install_${PROJECT_NAME} $(MAKE) install COMMENT "Installing ${PROJECT_NAME}")

