add_library(projections projection_flow.cc rhs.cc)
deal_ii_setup_target(projections)
target_link_libraries(projections flow base cutfem)

# Executables
add_executable(run-proj run.cc projection_flow.cc rhs.cc)
deal_ii_setup_target(run-proj)
target_link_libraries(run-proj projections flow base cutfem)

add_executable(convergence-proj convergence.cc)
deal_ii_setup_target(convergence-proj)
target_link_libraries(convergence-proj projections flow base cutfem)
