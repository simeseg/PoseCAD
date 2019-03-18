cmake_minimum_required(VERSION 2.8)

project(RealSense3Dmatching)

add_subdirectory(demo_src)
file(GLOB demo_src "*.h" "*.cpp")
add_executable(${PROJECT_NAME} ${demo_src})

find_package( PCL 1.0 REQUIRED)
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${PCL_LIBRARIES}  -lglfw -lGLU -lGL -lrealsense2)

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)

