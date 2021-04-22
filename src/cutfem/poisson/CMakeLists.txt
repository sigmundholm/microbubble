# add_library(poisson-nitsche poisson.cc rhs.cc)
# deal_ii_setup_target(poisson-nitsche)

add_executable(run-poisson-cutfem run.cc poisson.cc rhs.cc)
deal_ii_setup_target(run-poisson-cutfem)
target_link_libraries(run-poisson-cutfem cutfem)