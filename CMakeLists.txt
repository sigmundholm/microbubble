cmake_minimum_required(VERSION 2.8.12)

find_package(deal.II REQUIRED
        HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

deal_ii_initialize_cached_variables()

project(microbubble)

find_package(cutfem REQUIRED
        HINTS ${CUTFEM_DIR} $ENV{CUTFEM_DIR})


add_executable(nitsche-stokes src/stokes_channel_cylinder.cpp src/nitsche_stokes.cpp)
deal_ii_setup_target(nitsche-stokes)

add_executable(stokes-error src/nitsche_stokes.h src/nitsche_stokes.cpp src/stokes_error.cpp)
deal_ii_setup_target(stokes-error)

add_subdirectory(src)
