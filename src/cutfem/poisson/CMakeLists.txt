add_executable(run-poisson-cutfem run.cc poisson.cc rhs.cc)
deal_ii_setup_target(run-poisson-cutfem)
target_link_libraries(run-poisson-cutfem cutfem)

add_executable(convergence-poisson-cutfem convergence.cc poisson.cc rhs.cc)
deal_ii_setup_target(convergence-poisson-cutfem)
target_link_libraries(convergence-poisson-cutfem cutfem)

add_executable(condnum-poisson-cutfem condition_number.cc poisson.cc rhs.cc)
deal_ii_setup_target(condnum-poisson-cutfem)
target_link_libraries(condnum-poisson-cutfem cutfem)
