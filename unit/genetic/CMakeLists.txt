
add_executable(genetic test_genetic.cpp)
include($ENV{YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(genetic)
target_link_libraries(genetic portopt)

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME genetic_test COMMAND ${REAL_MPI_COMMAND} ./genetic)
else()
  add_test(NAME genetic_test COMMAND ./genetic)
endif()

