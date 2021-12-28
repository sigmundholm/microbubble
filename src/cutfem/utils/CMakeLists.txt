add_library(base cutfem_problem.cc utils.cc 
    stabilization/jump_stabilization.cc
    stabilization/face_selectors.cc
    stabilization/normal_derivative_computer.cc)
deal_ii_setup_target(base)
target_link_libraries(base)

add_library(scalar scalar_problem.cc cutfem_problem.cc)
deal_ii_setup_target(scalar)
target_link_libraries(scalar base)

add_library(flow flow_problem.cc cutfem_problem.cc)
deal_ii_setup_target(flow)
target_link_libraries(flow base)
