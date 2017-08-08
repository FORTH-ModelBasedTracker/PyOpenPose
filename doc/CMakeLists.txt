set(CMAKE_DOC_ROOT ${CMAKE_SOURCE_DIR}/PyOpenPoseLib)
mark_as_advanced(CMAKE_DOC_ROOT)
configure_file(docu.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating SDK documentation with Doxygen"
    VERBATIM)
