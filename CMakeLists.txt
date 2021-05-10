#############################################################################################
#
# Copyright (c) 2019 Jose Abel Ticona
# 
#############################################################################################


################# setup-Project ###################
cmake_minimum_required(VERSION 3.0)

## properties
set(VT_PROJECT_NAME       "HandTracking" )

## Options
OPTION(USE_DOUBLE_PRECISION "Use double precision"	OFF)# double precision
option(USE_CUDA "Use CUDA" OFF) # use cuda
option(VT_GEN_DOCS "Generate documentation" OFF)
####################################################


################# config-Project ###################

## Name of the project
project(${VT_PROJECT_NAME})

set( VT_PROJECT_PATH ${PROJECT_SOURCE_DIR} )

## Set module path to include cmake directory
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

## add file common generic
include(${VT_PROJECT_PATH}/cmake/common.cmake)
## add build external project
include(${VT_PROJECT_PATH}/cmake/build_external_project.cmake)

add_definitions(-DDATA_PATH_DEFINITION="${VT_PROJECT_PATH}/data/")# modify
add_definitions(-DPROJECT_PATH_DEFINITION="${VT_PROJECT_PATH}/")# modify

## use double precision
if (USE_DOUBLE_PRECISION)
  message(STATUS "Real = double")
	add_definitions( -DUSE_DOUBLE)
else()
  message(STATUS "Real = float")
endif (USE_DOUBLE_PRECISION)
## fin - use double precision

## use opencv
build_external_project(
	opencv 
	${VT_PROJECT_PATH}
	https://github.com/opencv/opencv.git 
	master
		-DBUILD_opencv_core=ON
		-DBUILD_opencv_highgui=ON
		-DBUILD_opencv_imgproc=ON
		-DBUILD_opencv_contrib=ON
		-DBUILD_DOCS:BOOL=FALSE
		-DBUILD_EXAMPLES:BOOL=FALSE
		-DBUILD_TESTS:BOOL=FALSE
		-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
		-DBUILD_SHARED_LIBS:BOOL=FALSE
		-DBUILD_NEW_PYTHON_SUPPORT:BOOL=OFF
		-DBUILD_WITH_DEBUG_INFO=OFF
		-DWITH_CUDA:BOOL=FALSE
		-DWITH_FFMPEG:BOOL=FALSE
		-DWITH_MSMF:BOOL=FALSE
		-DWITH_IPP:BOOL=FALSE
		-DBUILD_PERF_TESTS:BOOL=FALSE
		-DBUILD_PNG:BOOL=ON
		-DBUILD_JPEG:BOOL=ON
		-DBUILD_WITH_STATIC_CRT:BOOL=OFF
		-DBUILD_FAT_JAVA_LIB=OFF
)
## fin - use opencv

####################################################


################# structure-Project ###################

## add src
add_subdirectory(src) # modify

####################################################

set(EXT_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	set(EXT_CMAKE_BUILD_TYPE "Release")
endif()