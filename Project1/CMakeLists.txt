cmake_minimum_required(VERSION 3.0.0)
project(testrun VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(OpenCV_DIR "C:/opencv/build")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(imgDisplay imgDisplay.cpp)
target_link_libraries( imgDisplay ${OpenCV_LIBS} )

add_executable(vidDisplay vidDisplay.cpp filters.h filters.cpp)
target_link_libraries( vidDisplay ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
