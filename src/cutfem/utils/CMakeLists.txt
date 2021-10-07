add_library(base cutfem_problem.cc utils.cc)
deal_ii_setup_target(base)
target_link_libraries(base cutfem)

add_library(scalar scalar_problem.cc cutfem_problem.cc)
deal_ii_setup_target(scalar)
target_link_libraries(scalar base cutfem)
