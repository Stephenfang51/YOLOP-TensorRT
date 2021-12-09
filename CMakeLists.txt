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
include_directories(/home/yourpathto/cuda_11.1/include)
link_directories(/home/yourpathto/cuda_11.1/targets/x86_64-linux/lib)
include_directories(/home/yourpathto/TensorRT_installation/TensorRT-7.2.2.3/include)
link_directories(/home/yourpathto/TensorRT_installation/TensorRT-7.2.2.3/targets/x86_64-linux-gnu/lib)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)
set(OpenCV_DIR /home/yourpathto/TensorRT_installation/OpenCV/lib64/cmake/opencv4/) #incase you can't find OpenCV, manually set OpenCV_DIR


#-----OpenCV----#
# find_package(OpenCV PATHS /home/liwei.fang/TensorRT_installation/OpenCV)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

#link executable to libraries
add_executable(yolop ${PROJECT_SOURCE_DIR}/yolop.cpp)
target_link_libraries(yolop nvinfer)
target_link_libraries(yolop cudart)
target_link_libraries(yolop myplugins)
target_link_libraries(yolop ${OpenCV_LIBS})
target_link_libraries(yolop nvrtc)
add_definitions(-O0 -pthread)
