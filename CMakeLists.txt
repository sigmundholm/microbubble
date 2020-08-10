CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
find_package(deal.II REQUIRED
        HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(microbubble)

find_package(cutfem REQUIRED
        HINTS ${CUTFEM_DIR} $ENV{CUTFEM_DIR})

ADD_EXECUTABLE(nitsche-poisson src/nitsche_poisson.cpp)
DEAL_II_SETUP_TARGET(nitsche-poisson)

ADD_EXECUTABLE(nitsche-stokes src/stokes_channel_cylinder.cpp src/nitsche_stokes.cpp)
DEAL_II_SETUP_TARGET(nitsche-stokes)

ADD_EXECUTABLE(stokes-error src/nitsche_stokes.h src/nitsche_stokes.cpp src/stokes_error.cpp)
DEAL_II_SETUP_TARGET(stokes-error)

ADD_SUBDIRECTORY(src)
