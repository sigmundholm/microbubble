add_executable(heat-cutfem run.cc heat_eqn.cc rhs.cc)
deal_ii_setup_target(heat-cutfem)
target_link_libraries(heat-cutfem base scalar cutfem)

add_executable(convergence-heat-cutfem convergence.cc heat_eqn.cc rhs.cc)
deal_ii_setup_target(convergence-heat-cutfem)
target_link_libraries(convergence-heat-cutfem scalar)

add_executable(betatest-heat-cutfem beta_test.cc heat_eqn.cc rhs.cc)
deal_ii_setup_target(betatest-heat-cutfem)
target_link_libraries(betatest-heat-cutfem scalar)
