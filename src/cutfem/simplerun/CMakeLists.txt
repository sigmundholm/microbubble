add_executable(stokesCylinder simplerun.cc)
deal_ii_setup_target(stokesCylinder)
target_link_libraries(stokesCylinder stokes cutfem)

add_executable(time_dep time_dependent.cc)
deal_ii_setup_target(time_dep)
target_link_libraries(time_dep stokes cutfem)
