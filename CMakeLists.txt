cmake_minimum_required(VERSION 3.8)
project(cowpal)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CowInfo.msg"
)

# Install Python scripts (ROS nodes)
install(
  PROGRAMS
    scripts/websiteWriter.py
    scripts/calibration.py
    scripts/manager.py
    scripts/heightObtainor.py
    scripts/cowDetectionModel.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)


ament_package()
