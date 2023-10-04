cmake_minimum_required(VERSION 3.5.0)
project(testrun VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(OpenCV_DIR "C:/opencv/build")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(buildDb buildDb.cpp 
    csv_util.h csv_util.cpp 
    filters.h filters.cpp 
    utils.h utils.cpp 
    featureCalcs.h featureCalcs.cpp
)
target_link_libraries( buildDb ${OpenCV_LIBS} )

add_executable(searchImg searchImg.cpp 
    csv_util.h csv_util.cpp 
    filters.h filters.cpp 
    utils.h utils.cpp 
    featureCalcs.h featureCalcs.cpp
    distMetrics.h distMetrics.cpp
)
target_link_libraries( searchImg ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
