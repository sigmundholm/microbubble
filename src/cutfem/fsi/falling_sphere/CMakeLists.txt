
# Falling sphere
add_executable(fsi-falling-sphere run.cc falling_sphere.cc rhs.cc)
deal_ii_setup_target(fsi-falling-sphere)
target_link_libraries(fsi-falling-sphere navier-stokes)
