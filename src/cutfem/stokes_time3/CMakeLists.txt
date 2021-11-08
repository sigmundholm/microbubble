add_library(stokes-time3 stokes.cc rhs.cc)
deal_ii_setup_target(stokes-time3)
target_link_libraries(stokes-time3 flow base cutfem)

# Executables
add_executable(run-time3 run.cc stokes.cc rhs.cc)
deal_ii_setup_target(run-time3)
target_link_libraries(run-time3 stokes-time3 cutfem)

add_executable(convergence-time3 convergence.cc)
deal_ii_setup_target(convergence-time3)
target_link_libraries(convergence-time3 stokes-time3 cutfem)
