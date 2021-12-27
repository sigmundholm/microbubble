add_library(base cutfem_problem.cc utils.cc stabilization/jump_stabilization.cc)
deal_ii_setup_target(base)
target_link_libraries(base cutfem)

add_library(scalar scalar_problem.cc cutfem_problem.cc)
deal_ii_setup_target(scalar)
target_link_libraries(scalar base cutfem)

add_library(flow flow_problem.cc cutfem_problem.cc)
deal_ii_setup_target(flow)
target_link_libraries(flow base cutfem)
