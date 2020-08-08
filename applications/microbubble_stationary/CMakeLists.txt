add_library(stokes
        StokesCylinder.cc
        StokesRhs.cc)
DEAL_II_SETUP_TARGET(stokes)
target_link_libraries(stokes cutfem)

add_subdirectory(simplerun)
