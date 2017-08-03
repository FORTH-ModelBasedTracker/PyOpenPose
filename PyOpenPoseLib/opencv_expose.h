//
// Created by padeler on 3/8/2017.
//
/**
 * Code adapted for PyOpenPose from opencv cv2.cpp and pycompat.hpp for use with boost::python
 */

#ifndef PYOPENPOSE_OPENCV_EXPOSE_H_H
#define PYOPENPOSE_OPENCV_EXPOSE_H_H

#include <boost/python.hpp>


// Defines for Python 2/3 compatibility.
#ifndef __PYCOMPAT_HPP__
#define __PYCOMPAT_HPP__

#if PY_MAJOR_VERSION >= 3
// Python3 treats all ints as longs, PyInt_X functions have been removed.
#define PyInt_Check PyLong_Check
#define PyInt_CheckExact PyLong_CheckExact
#define PyInt_AsLong PyLong_AsLong
#define PyInt_AS_LONG PyLong_AS_LONG
#define PyInt_FromLong PyLong_FromLong
#define PyNumber_Int PyNumber_Long

// Python3 strings are unicode, these defines mimic the Python2 functionality.
#define PyString_Check PyUnicode_Check
#define PyString_FromString PyUnicode_FromString
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#define PyString_Size PyUnicode_GET_SIZE

// PyUnicode_AsUTF8 isn't available until Python 3.3
#if (PY_VERSION_HEX < 0x03030000)
#define PyString_AsString _PyUnicode_AsString
#else
#define PyString_AsString PyUnicode_AsUTF8
#endif
#endif

#endif // END HEADER GUARD



void expose_opencv();

#endif //PYOPENPOSE_OPENCV_EXPOSE_H_H
