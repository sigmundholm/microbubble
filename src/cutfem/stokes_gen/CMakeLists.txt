add_library(stokes_gen stokes_gen.cc rhs_gen.cc)
deal_ii_setup_target(stokes_gen)
target_link_libraries(stokes_gen cutfem)

# Executables
add_executable(run-gen run_gen.cc stokes_gen.cc rhs_gen.cc)
deal_ii_setup_target(run-gen)
target_link_libraries(run-gen stokes_gen cutfem)

add_executable(convergence-gen convergence.cc)
deal_ii_setup_target(convergence-gen)
target_link_libraries(convergence-gen stokes_gen cutfem)
