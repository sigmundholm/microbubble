CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

FIND_PACKAGE(deal.II 9.0.0 REQUIRED)

DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(microbubble)

ADD_EXECUTABLE(nitsche-poisson src/nitsche_poisson.cpp)
DEAL_II_SETUP_TARGET(nitsche-poisson)

ADD_EXECUTABLE(nitsche-stokes src/stokes_channel_cylinder.cpp src/nitsche_stokes.cpp)
DEAL_II_SETUP_TARGET(nitsche-stokes)

ADD_EXECUTABLE(stokes-error src/nitsche_stokes.h src/nitsche_stokes.cpp src/stokes_error.cpp)
DEAL_II_SETUP_TARGET(stokes-error)
