add_executable(convergence convergence.cpp)
deal_ii_setup_target(convergence)
target_link_libraries(convergence error stokes cutfem)
