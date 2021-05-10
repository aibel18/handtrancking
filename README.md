# HandTracking

tracking of the hand

The library was tested on Ubuntu 20.04.2 LTS <!-- Windows 10,  and Mac OS X 10.10.5. -->

**Author**: [Jose Abel Ticona](https://aibel18.github.io), **License**: MIT

## Implementation

## Build Instructions

- CMake 3.0
- OpenCV 4.5.2

### build in windows
	mkdir build
	cd build
	cmake ..
	MSBuild.exe HandTracking.sln /property:Configuration=Release or MSBuild.exe HandTracking.sln
	../bin/App

### build in linux
	mkdir build
	cd build
	cmake ..
	make
	../bin/App

Note: Please use a 64-bit target on a 64-bit operating system. 32-bit builds on a 64-bit OS are not supported.



