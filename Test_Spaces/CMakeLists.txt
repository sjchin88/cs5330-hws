cmake_minimum_required(VERSION 3.0.0)
project(testrun VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(OpenCV_DIR "C:/opencv/build")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(Testrun1 testrun.cpp)
add_executable(Tu_display_img tu_display_img.cpp)
target_link_libraries( Testrun1 ${OpenCV_LIBS} )
target_link_libraries( Tu_display_img ${OpenCV_LIBS} )

add_executable(Tu_video tu_video.cpp)
target_link_libraries( Tu_video ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
