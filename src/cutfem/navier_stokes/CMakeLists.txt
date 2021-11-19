add_library(navier-stokes navier_stokes.cc rhs.cc)
deal_ii_setup_target(navier-stokes)
target_link_libraries(navier-stokes stokes-time2)

# Executables
add_executable(run-ns run.cc navier_stokes.cc rhs.cc)
deal_ii_setup_target(run-ns)
target_link_libraries(run-ns navier-stokes)

add_executable(convergence-ns convergence.cc)
deal_ii_setup_target(convergence-ns)
target_link_libraries(convergence-ns navier-stokes)

add_executable(stat-conv-ns convergence_stationary.cc rhs_stat.cc)
deal_ii_setup_target(stat-conv-ns)
target_link_libraries(stat-conv-ns navier-stokes)

add_subdirectory(benchmarks)
