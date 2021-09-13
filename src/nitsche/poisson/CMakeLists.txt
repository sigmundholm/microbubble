add_library(poisson-nitsche poisson.cc rhs.cc)
deal_ii_setup_target(poisson-nitsche)

add_executable(run-poisson-nitsche run.cc poisson.cc rhs.cc)
deal_ii_setup_target(run-poisson-nitsche)
target_link_libraries(run-poisson-nitsche poisson-nitsche)

add_executable(convergence-poisson-nitsche convergence.cc poisson.cc rhs.cc)
deal_ii_setup_target(convergence-poisson-nitsche)
target_link_libraries(convergence-poisson-nitsche poisson-nitsche)
