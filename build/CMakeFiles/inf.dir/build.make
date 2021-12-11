# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake-3.22.0-rc2-linux-x86_64/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.22.0-rc2-linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/stephen/VScodeProjects/YOLOP-TensorRT

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/stephen/VScodeProjects/YOLOP-TensorRT/build

# Include any dependencies generated for this target.
include CMakeFiles/inf.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/inf.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/inf.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/inf.dir/flags.make

CMakeFiles/inf.dir/inference.cpp.o: CMakeFiles/inf.dir/flags.make
CMakeFiles/inf.dir/inference.cpp.o: ../inference.cpp
CMakeFiles/inf.dir/inference.cpp.o: CMakeFiles/inf.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/stephen/VScodeProjects/YOLOP-TensorRT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/inf.dir/inference.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/inf.dir/inference.cpp.o -MF CMakeFiles/inf.dir/inference.cpp.o.d -o CMakeFiles/inf.dir/inference.cpp.o -c /home/stephen/VScodeProjects/YOLOP-TensorRT/inference.cpp

CMakeFiles/inf.dir/inference.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inf.dir/inference.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/stephen/VScodeProjects/YOLOP-TensorRT/inference.cpp > CMakeFiles/inf.dir/inference.cpp.i

CMakeFiles/inf.dir/inference.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inf.dir/inference.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/stephen/VScodeProjects/YOLOP-TensorRT/inference.cpp -o CMakeFiles/inf.dir/inference.cpp.s

# Object files for target inf
inf_OBJECTS = \
"CMakeFiles/inf.dir/inference.cpp.o"

# External object files for target inf
inf_EXTERNAL_OBJECTS =

inf: CMakeFiles/inf.dir/inference.cpp.o
inf: CMakeFiles/inf.dir/build.make
inf: libmyplugins.so
inf: /home/stephen/TensorRT_installation/OpenCV/lib64/libopencv_dnn.so.4.3.0
inf: /home/stephen/TensorRT_installation/OpenCV/lib64/libopencv_highgui.so.4.3.0
inf: /home/stephen/TensorRT_installation/OpenCV/lib64/libopencv_ml.so.4.3.0
inf: /home/stephen/TensorRT_installation/OpenCV/lib64/libopencv_objdetect.so.4.3.0
inf: /home/stephen/TensorRT_installation/OpenCV/lib64/libopencv_photo.so.4.3.0
inf: /home/stephen/TensorRT_installation/OpenCV/lib64/libopencv_stitching.so.4.3.0
inf: /home/stephen/TensorRT_installation/OpenCV/lib64/libopencv_video.so.4.3.0
inf: /home/stephen/TensorRT_installation/OpenCV/lib64/libopencv_videoio.so.4.3.0
inf: /usr/local/cuda/lib64/libcudart.so
inf: /home/stephen/TensorRT_installation/OpenCV/lib64/libopencv_imgcodecs.so.4.3.0
inf: /home/stephen/TensorRT_installation/OpenCV/lib64/libopencv_calib3d.so.4.3.0
inf: /home/stephen/TensorRT_installation/OpenCV/lib64/libopencv_features2d.so.4.3.0
inf: /home/stephen/TensorRT_installation/OpenCV/lib64/libopencv_flann.so.4.3.0
inf: /home/stephen/TensorRT_installation/OpenCV/lib64/libopencv_imgproc.so.4.3.0
inf: /home/stephen/TensorRT_installation/OpenCV/lib64/libopencv_core.so.4.3.0
inf: CMakeFiles/inf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/stephen/VScodeProjects/YOLOP-TensorRT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable inf"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/inf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/inf.dir/build: inf
.PHONY : CMakeFiles/inf.dir/build

CMakeFiles/inf.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/inf.dir/cmake_clean.cmake
.PHONY : CMakeFiles/inf.dir/clean

CMakeFiles/inf.dir/depend:
	cd /home/stephen/VScodeProjects/YOLOP-TensorRT/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/stephen/VScodeProjects/YOLOP-TensorRT /home/stephen/VScodeProjects/YOLOP-TensorRT /home/stephen/VScodeProjects/YOLOP-TensorRT/build /home/stephen/VScodeProjects/YOLOP-TensorRT/build /home/stephen/VScodeProjects/YOLOP-TensorRT/build/CMakeFiles/inf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/inf.dir/depend
