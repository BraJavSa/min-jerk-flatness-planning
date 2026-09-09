cmake_minimum_required(VERSION 3.8)
project(min-jerk-flatness-planning)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Install real-time ROS 2 Python nodes
install(PROGRAMS
  case_1c/realtime/case_1_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME case_1
)

install(PROGRAMS
  case_2c/realtime/case_2_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME case_2
)

install(PROGRAMS
  case_3/realtime/caso3.py
  case_3/realtime/main_realtime.py
  DESTINATION lib/${PROJECT_NAME}
)

# Build C planners for Case 1 and Case 2
add_executable(case_1_planner
  case_1c/realtime/main.c
  case_1c/realtime/usv_params.c
  case_1c/realtime/min_jerk_qp.c
  case_1c/realtime/flatness_reconstruct.c
)
target_link_libraries(case_1_planner m)

add_executable(case_2_planner
  case_2c/realtime/main.c
  case_2c/realtime/usv_params.c
  case_2c/realtime/min_jerk_qp.c
  case_2c/realtime/sqp_optimizer.c
  case_2c/realtime/flatness_reconstruct.c
)
target_link_libraries(case_2_planner m)

install(TARGETS
  case_1_planner
  case_2_planner
  DESTINATION lib/${PROJECT_NAME}
)

# Install planning modules to share directory for clean import resolution
install(DIRECTORY
  case_1
  case_1c
  case_2
  case_2c
  case_3
  dynamic_model_iacquabot
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
