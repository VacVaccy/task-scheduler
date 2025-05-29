cmake_minimum_required(VERSION 3.10)
project(task_scheduler_cpp)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

configure_file(${CMAKE_SOURCE_DIR}/src/config.json ${CMAKE_BINARY_DIR}/config.json COPYONLY)

# Include directory
include_directories(include)

# Collect all cpp files in src/
file(GLOB SRC_FILES "src/*.cpp")

# Define the executable, output to src/scheduler
add_executable(scheduler ${SRC_FILES})

# Set the output directory for the executable
set_target_properties(scheduler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build"
)

