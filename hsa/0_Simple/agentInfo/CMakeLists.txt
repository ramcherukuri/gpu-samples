cmake_minimum_required(VERSION 3.10)

project(agentInfo)
# /opt/rocm/hsa/lib
# /opt/rocm/include/hsa
 
find_path(HSA_INCLUDES hsa.h HINTS /opt/rocm/include/hsa /opt/rocm/hsa/include PATH_SUFFIXES hsa)
find_library(HSA_LIBRARIES hsa-runtime64 HINTS /opt/rocm/lib /opt/rocm/hsa/lib)

# set(HSA_INCLUDES ${FIND_HSA_INCLUDES})
# set(HSA_LIBRARIES ${FIND_HSA_LIBRARIES})

set(TEST_SOURCE_FILES agentInfo.cpp)
set(CMAKE_INCLUDE_PATH ${HSA_INCLUDES})

# include_directories(${FIND_HSA_INCLUDES})
include_directories(${HSA_INCLUDES})

add_executable(agentInfo ${TEST_SOURCE_FILES})

target_link_libraries(agentInfo ${HSA_LIBRARIES} )


