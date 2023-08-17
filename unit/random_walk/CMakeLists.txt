
add_executable(random_walk test_random_walk.cpp)
include($ENV{YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(random_walk)
target_link_libraries(random_walk portopt)

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME random_walk_test COMMAND ${REAL_MPI_COMMAND} ./random_walk)
else()
  add_test(NAME random_walk_test COMMAND ./random_walk)
endif()

