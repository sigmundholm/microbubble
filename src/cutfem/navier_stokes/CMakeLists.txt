add_library(navier-stokes navier_stokes.cc rhs.cc)
deal_ii_setup_target(navier-stokes)
target_link_libraries(navier-stokes base flow stokes-time2 cutfem)

# Executables
add_executable(run-ns run.cc navier_stokes.cc rhs.cc)
deal_ii_setup_target(run-ns)
target_link_libraries(run-ns stokes-time2 flow base cutfem)

add_executable(convergence-ns convergence.cc)
deal_ii_setup_target(convergence-ns)
target_link_libraries(convergence-ns navier-stokes stokes-time2 flow base cutfem)
