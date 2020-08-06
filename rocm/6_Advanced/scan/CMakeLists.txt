cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

project(scan)

find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()

set(TEST_SOURCE_FILES scan.cpp)

set(TEST_TARGET_NAME scan)
set(CMAKE_INCLUDE_PATH ../../Common/)

set(COMMON_DIRECTORY ../../Common/)

set_source_files_properties(${TEST_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${TEST_TARGET_NAME} ${TEST_SOURCE_FILES})

target_include_directories( ${TEST_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${COMMON_DIRECTORY}>)

