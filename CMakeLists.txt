cmake_minimum_required(VERSION 3.19)

project(yolovp)
add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

#CUDA
find_package(CUDA REQUIRED)


# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/targets/x86_64-linux/lib)
include_directories(/home/yourpath/TensorRT_installation/TensorRT-7.2.2.3/include)
link_directories(/home/yourpath/TensorRT_installation/TensorRT-7.2.2.3/targets/x86_64-linux-gnu/lib)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/src/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

set(OpenCV_DIR /home/yourpath/TensorRT_installation/OpenCV/lib64/cmake/opencv4/) #incase you can't find OpenCV, manually set OpenCV_DIR

#-----OpenCV----#
# find_package(OpenCV PATHS /home/yourpath/TensorRT_installation/OpenCV)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

#link executable to libraries
add_executable(inf ${PROJECT_SOURCE_DIR}/inference.cpp)
add_executable(buildengine ${PROJECT_SOURCE_DIR}/buildEngine.cpp)
target_link_libraries(inf nvinfer)
target_link_libraries(inf cudart)
target_link_libraries(inf myplugins)
target_link_libraries(inf ${OpenCV_LIBS})
target_link_libraries(inf nvrtc)
target_link_libraries(buildengine nvinfer)
target_link_libraries(buildengine cudart)
target_link_libraries(buildengine myplugins)
target_link_libraries(buildengine ${OpenCV_LIBS})
target_link_libraries(buildengine nvrtc)
add_definitions(-O0 -pthread)