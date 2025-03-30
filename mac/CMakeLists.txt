cmake_minimum_required(VERSION 3.17)
project(TFLiteCheck)

# Definindo o padrão do C++ para 14
set(CMAKE_CXX_STANDARD 14)

# TensorFlow Lite - includes e lib
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/include/)
add_library(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/mac/libtensorflowlite.dylib)

# 🧠 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add image utils
set(SOURCES
    main.cpp
    imageUtils.cpp   
)

add_executable(TFLiteCheck ${SOURCES})

# 🔗 Link to TensorFlow Lite and OpenCV
target_link_libraries(TFLiteCheck tensorflowlite ${OpenCV_LIBS})
