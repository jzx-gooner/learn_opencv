cmake_minimum_required(VERSION 3.14)
project(k_means)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


add_executable(k_means main.cpp)

include_directories(
        include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}
)

target_link_libraries(k_means ${OpenCV_LIBS})