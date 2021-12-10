
# Moving sphere
add_executable(fsi-moving-sphere moving_sphere.cc rhs.cc)
deal_ii_setup_target(fsi-moving-sphere)
target_link_libraries(fsi-moving-sphere navier-stokes)
