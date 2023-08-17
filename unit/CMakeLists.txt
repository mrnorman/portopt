cmake_minimum_required(VERSION 3.0)
project(portopt_unit)
enable_testing()

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

# YAKL library
add_subdirectory($ENV{YAKL_HOME} ./yakl )
add_subdirectory(..              ./portopt)

# Unit test directories
add_subdirectory(adam)
add_subdirectory(random_walk)
add_subdirectory(genetic)

