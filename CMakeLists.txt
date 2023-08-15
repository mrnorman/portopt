cmake_minimum_required(VERSION 3.0)
project(portopt)

add_library(portopt INTERFACE)
target_link_libraries(portopt INTERFACE yakl)
target_include_directories(portopt INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

