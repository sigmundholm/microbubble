add_library(error
        ErrorStokesCylinder.cc
        ErrorRhs.cc)
deal_ii_setup_target(error)
target_link_libraries(error stokes cutfem)

add_executable(error_run error.cpp ErrorStokesCylinder.cc ErrorRhs.cc)
deal_ii_setup_target(error_run)
target_link_libraries(error_run stokes cutfem)
