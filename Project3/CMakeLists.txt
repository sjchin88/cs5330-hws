cmake_minimum_required(VERSION 3.0.0)
project(testrun VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(OpenCV_DIR "C:/opencv/build")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(buildDb buildDb.cpp 
    file_util.h file_util.cpp
    filters.h filters.cpp 
    features.h features.cpp
)
target_link_libraries( buildDb ${OpenCV_LIBS} )

add_executable(vidDisplay vidDisplay.cpp 
    file_util.h file_util.cpp 
    filters.h filters.cpp
    features.h features.cpp
    classify.h classify.cpp
)
target_link_libraries( vidDisplay ${OpenCV_LIBS} )

add_executable(imgDisplay imgDisplay.cpp 
    file_util.h file_util.cpp
    filters.h filters.cpp
    features.h features.cpp
    classify.h classify.cpp
)
target_link_libraries( imgDisplay ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
