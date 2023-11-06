cmake_minimum_required(VERSION 3.0.0)
project(testrun VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(OpenCV_DIR "C:/opencv/build")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable(calibrateCamera calibrateCamera.cpp 
    utils.h utils.cpp
)
target_link_libraries( calibrateCamera ${OpenCV_LIBS} )

add_executable(displayAR displayAR.cpp 
    utils.h utils.cpp objLoader.h
)
target_link_libraries( displayAR ${OpenCV_LIBS} )

add_executable(harrisCorner harrisCorner.cpp 
    utils.h utils.cpp objLoader.h
)
target_link_libraries( harrisCorner ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
