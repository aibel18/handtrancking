##########################################################################################
#
# Copyright (c) 2019 Jose Abel Ticona
#
# 
##########################################################################################

# add opencv
set(BUILD_SHARED_LIBS OFF)
find_package(OpenCV REQUIRED PATHS ${VT_PROJECT_PATH}/build/extern/opencv NO_DEFAULT_PATH)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# create my executable
add_executable (App Main.cpp
	ProcessingImage.h
	stereovision.h)

target_include_directories(App PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(App PRIVATE ${OpenCV_LIBS})