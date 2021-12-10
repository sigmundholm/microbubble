# Benchmark tests
add_library(bm-ns ns_benchmark.cc)
deal_ii_setup_target(bm-ns)
target_link_libraries(bm-ns navier-stokes)

add_executable(bm-2d-1 stationary_2D-1.cc)
deal_ii_setup_target(bm-2d-1)
target_link_libraries(bm-2d-1 bm-ns)

add_executable(bm-2d-3 timedep_2D-3.cc)
deal_ii_setup_target(bm-2d-3)
target_link_libraries(bm-2d-3 bm-ns)
