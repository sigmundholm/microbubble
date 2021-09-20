add_library(flow flow_problem.cc)
deal_ii_setup_target(flow)
target_link_libraries(flow cutfem)
