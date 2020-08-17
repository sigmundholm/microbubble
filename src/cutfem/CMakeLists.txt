add_library(stokes
        StokesCylinder.cc
        StokesRhs.cc)
deal_ii_setup_target(stokes)
target_link_libraries(stokes cutfem)

add_subdirectory(simplerun)
add_subdirectory(error)
add_subdirectory(convergence)
