cmake_minimum_required(VERSION 3.7)
project(learn_opencv) #将untitled改为自己的工程名
set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cpp)
add_executable(untitled ${SOURCE_FILES})
target_link_libraries( untitled ${OpenCV_LIBS} )