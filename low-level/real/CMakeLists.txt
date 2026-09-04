cmake_minimum_required(VERSION 3.10)
project(b2z1_real)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
  rospy
  sensor_msgs
  std_msgs
  std_srvs
)

catkin_python_setup()

add_message_files(
  FILES
  B2LowCommand.msg
  B2LowState.msg
  Z1Command.msg
  Z1State.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  geometry_msgs
  message_runtime
  roscpp
  rospy
  sensor_msgs
  std_msgs
  std_srvs
)

catkin_install_python(PROGRAMS
  scripts/b2z1_policy_node.py
  scripts/export_b2z1_policy.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES README.md DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

set(UNITREE_SDK2_ROOT "" CACHE PATH "Path to the official unitree_sdk2 checkout")
if(UNITREE_SDK2_ROOT)
  set(UNITREE_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
  set(UNITREE_SDK2_LIB "${UNITREE_SDK2_ROOT}/lib/${UNITREE_ARCH}/libunitree_sdk2.a")
  set(UNITREE_DDS_LIB_DIR "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH}")
  if(NOT EXISTS "${UNITREE_SDK2_LIB}")
    message(FATAL_ERROR "unitree_sdk2 library not found: ${UNITREE_SDK2_LIB}")
  endif()

  add_executable(b2_sdk2_bridge src/b2_sdk2_bridge.cpp)
  add_dependencies(b2_sdk2_bridge ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_compile_features(b2_sdk2_bridge PRIVATE cxx_std_17)
  target_include_directories(b2_sdk2_bridge PRIVATE
    ${catkin_INCLUDE_DIRS}
    "${UNITREE_SDK2_ROOT}/include"
    "${UNITREE_SDK2_ROOT}/thirdparty/include"
    "${UNITREE_SDK2_ROOT}/thirdparty/include/ddscxx"
  )
  target_link_libraries(b2_sdk2_bridge
    ${catkin_LIBRARIES}
    "${UNITREE_SDK2_LIB}"
    "${UNITREE_DDS_LIB_DIR}/libddscxx.so"
    "${UNITREE_DDS_LIB_DIR}/libddsc.so"
    pthread
  )
  set_target_properties(b2_sdk2_bridge PROPERTIES
    BUILD_RPATH "${UNITREE_DDS_LIB_DIR}"
    INSTALL_RPATH "${UNITREE_DDS_LIB_DIR}"
  )
  install(TARGETS b2_sdk2_bridge
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
else()
  message(STATUS "UNITREE_SDK2_ROOT is empty; b2_sdk2_bridge will not be built")
endif()

set(Z1_SDK_ROOT "" CACHE PATH "Path to the official z1_sdk checkout")
if(Z1_SDK_ROOT)
  set(Z1_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
  set(Z1_SDK_LIB "${Z1_SDK_ROOT}/lib/libZ1_SDK_${Z1_ARCH}.so")
  if(NOT EXISTS "${Z1_SDK_LIB}")
    message(FATAL_ERROR "Z1 SDK library not found: ${Z1_SDK_LIB}")
  endif()

  find_package(Eigen3 REQUIRED)
  add_executable(z1_sdk_bridge src/z1_sdk_bridge.cpp)
  add_dependencies(z1_sdk_bridge ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_compile_features(z1_sdk_bridge PRIVATE cxx_std_14)
  target_include_directories(z1_sdk_bridge PRIVATE
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    "${Z1_SDK_ROOT}/include"
  )
  target_link_libraries(z1_sdk_bridge
    ${catkin_LIBRARIES}
    "${Z1_SDK_LIB}"
    pthread
  )
  set_target_properties(z1_sdk_bridge PROPERTIES
    BUILD_RPATH "${Z1_SDK_ROOT}/lib"
    INSTALL_RPATH "${Z1_SDK_ROOT}/lib"
  )
  install(TARGETS z1_sdk_bridge
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
else()
  message(STATUS "Z1_SDK_ROOT is empty; z1_sdk_bridge will not be built")
endif()
