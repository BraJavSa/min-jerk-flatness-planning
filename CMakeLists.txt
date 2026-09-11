cmake_minimum_required(VERSION 3.8)
project(min-jerk-flatness-planning)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(casadi REQUIRED)

install(PROGRAMS
  Mass_Symmetry/RealtimeController/planner_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME mass_symmetry_planner
)
install(PROGRAMS
  Mass_Symmetry/RealtimeController/planner_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME case1_mpc_planner
)
install(PROGRAMS
  Mass_Symmetry/RealtimeController/recorder_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME mass_symmetry_recorder
)
install(PROGRAMS
  Mass_Symmetry/RealtimeController/recorder_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME case1_mpc_recorder
)
install(PROGRAMS
  Mass_Symmetry/RealtimeController/plot_realtime_results.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME mass_symmetry_plot
)
install(PROGRAMS
  Mass_Symmetry/RealtimeController/plot_realtime_results.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME case1_mpc_plot
)

install(PROGRAMS
  Pseudo-Flat_Reconstruction/RealtimeController/planner_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME pseudo_flat_planner
)
install(PROGRAMS
  Pseudo-Flat_Reconstruction/RealtimeController/planner_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME case2_mpc_planner
)
install(PROGRAMS
  Pseudo-Flat_Reconstruction/RealtimeController/recorder_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME pseudo_flat_recorder
)
install(PROGRAMS
  Pseudo-Flat_Reconstruction/RealtimeController/recorder_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME case2_mpc_recorder
)
install(PROGRAMS
  Pseudo-Flat_Reconstruction/RealtimeController/plot_realtime_results.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME pseudo_flat_plot
)
install(PROGRAMS
  Pseudo-Flat_Reconstruction/RealtimeController/plot_realtime_results.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME case2_mpc_plot
)

install(PROGRAMS
  Fictitious-Input_Full_Actuation/RealtimeController/planner_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fictitious_input_planner
)
install(PROGRAMS
  Fictitious-Input_Full_Actuation/RealtimeController/planner_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME case3_mpc_planner
)
install(PROGRAMS
  Fictitious-Input_Full_Actuation/RealtimeController/mpc_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fictitious_input_mpc_node
)
install(PROGRAMS
  Fictitious-Input_Full_Actuation/RealtimeController/mpc_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME case3_mpc_node
)
install(PROGRAMS
  Fictitious-Input_Full_Actuation/RealtimeController/recorder_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fictitious_input_recorder
)
install(PROGRAMS
  Fictitious-Input_Full_Actuation/RealtimeController/recorder_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME case3_mpc_recorder
)
install(PROGRAMS
  Fictitious-Input_Full_Actuation/RealtimeController/plot_realtime_results.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fictitious_input_plot
)
install(PROGRAMS
  Fictitious-Input_Full_Actuation/RealtimeController/plot_realtime_results.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME case3_mpc_plot
)

add_executable(mass_symmetry_c_planner
  Mass_Symmetry/RealtimeController/main.c
  Mass_Symmetry/RealtimeController/usv_params.c
  Mass_Symmetry/RealtimeController/min_jerk_qp.c
  Mass_Symmetry/RealtimeController/flatness_reconstruct.c
)
target_link_libraries(mass_symmetry_c_planner m)

add_executable(case1_mpc_c_planner
  Mass_Symmetry/RealtimeController/main.c
  Mass_Symmetry/RealtimeController/usv_params.c
  Mass_Symmetry/RealtimeController/min_jerk_qp.c
  Mass_Symmetry/RealtimeController/flatness_reconstruct.c
)
target_link_libraries(case1_mpc_c_planner m)

add_executable(pseudo_flat_c_planner
  Pseudo-Flat_Reconstruction/RealtimeController/main.c
  Pseudo-Flat_Reconstruction/RealtimeController/usv_params.c
  Pseudo-Flat_Reconstruction/RealtimeController/min_jerk_qp.c
  Pseudo-Flat_Reconstruction/RealtimeController/sqp_optimizer.c
  Pseudo-Flat_Reconstruction/RealtimeController/flatness_reconstruct.c
)
target_link_libraries(pseudo_flat_c_planner m)

add_executable(case2_mpc_c_planner
  Pseudo-Flat_Reconstruction/RealtimeController/main.c
  Pseudo-Flat_Reconstruction/RealtimeController/usv_params.c
  Pseudo-Flat_Reconstruction/RealtimeController/min_jerk_qp.c
  Pseudo-Flat_Reconstruction/RealtimeController/sqp_optimizer.c
  Pseudo-Flat_Reconstruction/RealtimeController/flatness_reconstruct.c
)
target_link_libraries(case2_mpc_c_planner m)

add_executable(mass_symmetry_mpc_node
  Mass_Symmetry/RealtimeController/mpc_node.cpp
  Mass_Symmetry/RealtimeController/nmpc_flatness.cpp
  Mass_Symmetry/RealtimeController/usv_params.c
  Mass_Symmetry/RealtimeController/flatness_reconstruct.c
)
target_include_directories(mass_symmetry_mpc_node PRIVATE Mass_Symmetry/RealtimeController)
ament_target_dependencies(mass_symmetry_mpc_node rclcpp std_msgs nav_msgs)
target_link_libraries(mass_symmetry_mpc_node casadi m)
target_compile_features(mass_symmetry_mpc_node PRIVATE cxx_std_17)

add_executable(case1_mpc_node
  Mass_Symmetry/RealtimeController/mpc_node.cpp
  Mass_Symmetry/RealtimeController/nmpc_flatness.cpp
  Mass_Symmetry/RealtimeController/usv_params.c
  Mass_Symmetry/RealtimeController/flatness_reconstruct.c
)
target_include_directories(case1_mpc_node PRIVATE Mass_Symmetry/RealtimeController)
ament_target_dependencies(case1_mpc_node rclcpp std_msgs nav_msgs)
target_link_libraries(case1_mpc_node casadi m)
target_compile_features(case1_mpc_node PRIVATE cxx_std_17)

add_executable(pseudo_flat_mpc_node
  Pseudo-Flat_Reconstruction/RealtimeController/mpc_node.cpp
  Pseudo-Flat_Reconstruction/RealtimeController/nmpc_flatness.cpp
  Pseudo-Flat_Reconstruction/RealtimeController/usv_params.c
  Pseudo-Flat_Reconstruction/RealtimeController/flatness_reconstruct.c
  Pseudo-Flat_Reconstruction/RealtimeController/sqp_optimizer.c
)
target_include_directories(pseudo_flat_mpc_node PRIVATE Pseudo-Flat_Reconstruction/RealtimeController)
ament_target_dependencies(pseudo_flat_mpc_node rclcpp std_msgs nav_msgs)
target_link_libraries(pseudo_flat_mpc_node casadi m)
target_compile_features(pseudo_flat_mpc_node PRIVATE cxx_std_17)

add_executable(case2_mpc_node
  Pseudo-Flat_Reconstruction/RealtimeController/mpc_node.cpp
  Pseudo-Flat_Reconstruction/RealtimeController/nmpc_flatness.cpp
  Pseudo-Flat_Reconstruction/RealtimeController/usv_params.c
  Pseudo-Flat_Reconstruction/RealtimeController/flatness_reconstruct.c
  Pseudo-Flat_Reconstruction/RealtimeController/sqp_optimizer.c
)
target_include_directories(case2_mpc_node PRIVATE Pseudo-Flat_Reconstruction/RealtimeController)
ament_target_dependencies(case2_mpc_node rclcpp std_msgs nav_msgs)
target_link_libraries(case2_mpc_node casadi m)
target_compile_features(case2_mpc_node PRIVATE cxx_std_17)

install(TARGETS
  mass_symmetry_c_planner
  case1_mpc_c_planner
  pseudo_flat_c_planner
  case2_mpc_c_planner
  mass_symmetry_mpc_node
  case1_mpc_node
  pseudo_flat_mpc_node
  case2_mpc_node
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  Mass_Symmetry/RealtimeController/main.py
  DESTINATION share/${PROJECT_NAME}/launch
  RENAME mass_symmetry.launch.py
)
install(FILES
  Mass_Symmetry/RealtimeController/main.py
  DESTINATION share/${PROJECT_NAME}/launch
  RENAME case1_mpc.launch.py
)
install(FILES
  Pseudo-Flat_Reconstruction/RealtimeController/main.py
  DESTINATION share/${PROJECT_NAME}/launch
  RENAME pseudo_flat.launch.py
)
install(FILES
  Pseudo-Flat_Reconstruction/RealtimeController/main.py
  DESTINATION share/${PROJECT_NAME}/launch
  RENAME case2_mpc.launch.py
)
install(FILES
  Fictitious-Input_Full_Actuation/RealtimeController/main.py
  DESTINATION share/${PROJECT_NAME}/launch
  RENAME fictitious_input.launch.py
)
install(FILES
  Fictitious-Input_Full_Actuation/RealtimeController/main.py
  DESTINATION share/${PROJECT_NAME}/launch
  RENAME case3_mpc.launch.py
)

install(DIRECTORY
  Mass_Symmetry
  Pseudo-Flat_Reconstruction
  Fictitious-Input_Full_Actuation
  dynamic_model_iacquabot
  comparison
  DESTINATION share/${PROJECT_NAME}
  PATTERN "*.o" EXCLUDE
  PATTERN "*.a" EXCLUDE
)

ament_package()
