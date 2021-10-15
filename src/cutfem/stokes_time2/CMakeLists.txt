add_library(stokes-time2 stokes.cc rhs.cc)
deal_ii_setup_target(stokes-time2)
target_link_libraries(stokes-time2 base flow cutfem)

# Executables
add_executable(run-time2 run.cc stokes.cc rhs.cc)
deal_ii_setup_target(run-time2)
target_link_libraries(run-time2 stokes-time2 projections flow base cutfem)

add_executable(convergence-time2 convergence.cc)
deal_ii_setup_target(convergence-time2)
target_link_libraries(convergence-time2 stokes-time2 stokes-time projections flow base cutfem)

# Moving ball example
add_executable(ex1-moving-ball ex_moving_ball_run.cc ex_moving_ball.cc)
deal_ii_setup_target(ex1-moving-ball)
target_link_libraries(ex1-moving-ball stokes-time2 stokes-time projections flow base cutfem)
