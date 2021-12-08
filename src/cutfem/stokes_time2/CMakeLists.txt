add_library(stokes-time2 stokes.cc rhs.cc)
deal_ii_setup_target(stokes-time2)
target_link_libraries(stokes-time2 base flow cutfem)

# Executables
add_executable(run-time2 run.cc stokes.cc rhs.cc)
deal_ii_setup_target(run-time2)
target_link_libraries(run-time2 stokes-time2 projections flow base cutfem)

add_executable(convergence-time2 convergence.cc)
deal_ii_setup_target(convergence-time2)
target_link_libraries(convergence-time2 stokes-time2 stokes-time projections flow base cutfem)

add_executable(stat-convergence convergence_stationary.cc rhs_stat.cc)
deal_ii_setup_target(stat-convergence)
target_link_libraries(stat-convergence stokes-time2)
