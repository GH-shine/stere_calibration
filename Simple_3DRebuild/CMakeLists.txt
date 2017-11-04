cmake_minimum_required( VERSION 2.8)

project(Simple_3DRebuild)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Simple_3DRebuild main.cpp)

add_library(Camera_Cailb_LIB Camera_Cailb.cpp)

target_link_libraries(Simple_3DRebuild ${OpenCV_LIBS})
target_link_libraries(Simple_3DRebuild Camera_Cailb_LIB)

install(TARGETS Simple_3DRebuild RUNTIME DESTINATION bin)
