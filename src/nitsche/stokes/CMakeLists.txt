add_executable(nitsche-stokes stokes_channel_cylinder.cpp nitsche_stokes.cpp)
deal_ii_setup_target(nitsche-stokes)

add_executable(stokes-error nitsche_stokes.h nitsche_stokes.cpp stokes_error.cpp)
deal_ii_setup_target(stokes-error)
