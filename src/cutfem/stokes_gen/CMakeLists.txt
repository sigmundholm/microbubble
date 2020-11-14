add_library(stokes_gen stokes_gen.cc rhs_gen.cc)
deal_ii_setup_target(stokes_gen)
target_link_libraries(stokes_gen cutfem)

# Executables
add_executable(run_gen run_gen.cc stokes_gen.cc rhs_gen.cc)
deal_ii_setup_target(run_gen)
target_link_libraries(run_gen stokes_gen cutfem)
