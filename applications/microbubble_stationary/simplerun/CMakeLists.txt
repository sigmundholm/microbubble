add_executable(stokesCylinder
        simplerun.cc)

DEAL_II_SETUP_TARGET(stokesCylinder)
target_link_libraries(stokesCylinder stokes cutfem)
