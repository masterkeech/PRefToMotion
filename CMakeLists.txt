cmake_minimum_required(VERSION 3.11)

project(PRefToMotion)

if (NOT DEFINED NUKE_VERSION)
    set(NUKE_VERSION 11.3v6)
endif()

set(CMAKE_CXX_STANDARD 11)

SET(CMAKE_SHARED_LIBRARY_PREFIX "")

if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
    SET(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")
endif()

if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
    set(NDKDIR /Applications/Nuke${NUKE_VERSION}/Nuke${NUKE_VERSION}.app/Contents/MacOS)
endif ()

link_directories(${NDKDIR} ${NDKDIR}/../Frameworks)
include_directories(${NDKDIR}/include headers)

add_subdirectory(src)
add_subdirectory(gizmos)
