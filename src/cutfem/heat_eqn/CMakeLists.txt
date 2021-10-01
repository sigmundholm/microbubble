add_executable(heat-cutfem run.cc heat_eqn.cc rhs.cc)
deal_ii_setup_target(heat-cutfem)
target_link_libraries(heat-cutfem base scalar cutfem)

add_executable(convergence-heat-cutfem convergence.cc heat_eqn.cc rhs.cc)
deal_ii_setup_target(convergence-heat-cutfem)
target_link_libraries(convergence-heat-cutfem scalar base cutfem)
