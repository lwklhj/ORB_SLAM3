cmake_minimum_required(VERSION 2.8)
project(DBoW3)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

set(HDRS_DBOW3
  src/BowVector.h
  src/Database.h
  src/DBoW3.h
  src/DescManip.h
  src/exports.h
  src/FeatureVector.h
  src/QueryResults.h
  src/quicklz.h
  src/ScoringObject.h
  src/timers.h
  src/Vocabulary.h)
set(SRCS_DBOW3
  src/BowVector.cpp
  src/DescManip.cpp
  src/quicklz.c
  src/FeatureVector.cpp
  src/ScoringObject.cpp
  src/Database.cpp
  src/QueryResults.cpp
  src/Vocabulary.cpp)

set(HDRS_DUTILS
  DUtils/Random.h
  DUtils/Timestamp.h)
set(SRCS_DUTILS
  DUtils/Random.cpp
  DUtils/Timestamp.cpp)


find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 3.0 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 3.0 not found.")
   endif()
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${OpenCV_INCLUDE_DIRS})
add_library(DBoW3 SHARED ${SRCS_DBOW3} ${SRCS_DUTILS})
target_link_libraries(DBoW3 ${OpenCV_LIBS})