cmake_minimum_required(VERSION 3.17)
project(TFLiteCheck)

set(CMAKE_CXX_STANDARD 14)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/include/)

find_library(TFLITE tensorflowlite.dll.if HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/windows_x64/)
find_library(FLATBUFFERS flatbuffers  HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/windows_x64/)


# 🧠 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add image utils
set(SOURCES
    main.cpp
    imageUtils.cpp   
)

add_executable(TFLiteCheck ${SOURCES})
target_link_libraries(TFLiteCheck PRIVATE ${TFLITE} ${FLATBUFFERS})

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/windows_x64/tensorflowlite.dll ${CMAKE_BINARY_DIR}/Release/
)