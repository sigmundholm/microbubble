add_library(scalar scalar_problem.cc)
deal_ii_setup_target(scalar)
target_link_libraries(scalar cutfem)
