add_library(stokes-time stokes_gen.cc rhs_gen.cc)
deal_ii_setup_target(stokes-time)
target_link_libraries(stokes-time cutfem)

# Executables
add_executable(run-time run_gen.cc stokes_gen.cc rhs_gen.cc)
deal_ii_setup_target(run-time)
target_link_libraries(run-time stokes-time cutfem)

add_executable(convergence-time convergence.cc)
deal_ii_setup_target(convergence-time)
target_link_libraries(convergence-time stokes-time cutfem)
