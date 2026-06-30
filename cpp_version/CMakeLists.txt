cmake_minimum_required(VERSION 3.16)
project(VideoDetection VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Find dependencies ----
# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")

# FFmpeg components
set(FFMPEG_COMPONENTS
    avformat
    avcodec
    avutil
    swscale
    avdevice
    avfilter
)

foreach(comp ${FFMPEG_COMPONENTS})
    find_library(FFMPEG_${comp}_LIBRARY ${comp})
    if(FFMPEG_${comp}_LIBRARY)
        message(STATUS "Found lib${comp}: ${FFMPEG_${comp}_LIBRARY}")
    else()
        message(WARNING "lib${comp} not found")
    endif()
endforeach()

# Determine FFmpeg include dirs
set(FFMPEG_INCLUDE_DIRS "")
foreach(dir
    /usr/include
    /usr/include/x86_64-linux-gnu
    /usr/local/include
    ${PROJECT_SOURCE_DIR}/../third_party/ffmpeg/include
    ${CMAKE_SOURCE_DIR}/third_party/ffmpeg/include
)
    if(EXISTS "${dir}/libavformat/avformat.h")
        list(APPEND FFMPEG_INCLUDE_DIRS ${dir})
        message(STATUS "FFmpeg headers found: ${dir}")
        break()
    endif()
endforeach()

if(NOT FFMPEG_INCLUDE_DIRS)
    message(WARNING "FFmpeg headers not found. Set FFMPEG_INCLUDE_DIRS manually.")
endif()

# ONNX Runtime - search downloaded SDK first
set(ONNXRUNTIME_INCLUDE_DIR "")
set(ONNXRUNTIME_LIB_DIR "")

# Search paths for ONNX Runtime C++ SDK
set(ORT_SEARCH_PATHS
    ${CMAKE_SOURCE_DIR}/../third_party/onnxruntime
    ${CMAKE_SOURCE_DIR}/third_party/onnxruntime
    $ENV{HOME}/onnxruntime
    /usr
    /usr/local
)

foreach(search_path ${ORT_SEARCH_PATHS})
    if(EXISTS "${search_path}/include/onnxruntime_cxx_api.h")
        set(ONNXRUNTIME_INCLUDE_DIR "${search_path}/include")
        if(EXISTS "${search_path}/lib/libonnxruntime.so")
            set(ONNXRUNTIME_LIB_DIR "${search_path}/lib")
        endif()
        break()
    endif()
endforeach()

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB_DIR)
    message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime lib:     ${ONNXRUNTIME_LIB_DIR}")
else()
    message(WARNING "ONNX Runtime C++ SDK not found.")
    message(WARNING "Run scripts/setup_cpp_deps.sh to download dependencies.")
endif()

# ---- Source files ----
set(SOURCES
    src/main.cpp
    src/video_decoder.cpp
    src/preprocessor.cpp
    src/inference_onnx.cpp
    src/postprocessor.cpp
    src/video_encoder.cpp
)

# ---- Executable ----
add_executable(video_detection ${SOURCES})

target_include_directories(video_detection PRIVATE
    src/
    ${OpenCV_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(video_detection PRIVATE
    ${OpenCV_LIBS}
)

# Link FFmpeg libraries
foreach(comp ${FFMPEG_COMPONENTS})
    if(FFMPEG_${comp}_LIBRARY)
        target_link_libraries(video_detection PRIVATE ${FFMPEG_${comp}_LIBRARY})
    endif()
endforeach()

# Link ONNX Runtime
if(DEFINED ONNXRUNTIME_LIB_DIR)
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_LIB_DIR} NO_DEFAULT_PATH)
    if(ONNXRUNTIME_LIB)
        target_link_libraries(video_detection PRIVATE ${ONNXRUNTIME_LIB})
    endif()
endif()

# pthread (required by FFmpeg and ORT)
find_package(Threads REQUIRED)
target_link_libraries(video_detection PRIVATE Threads::Threads)

# Link system libs
target_link_libraries(video_detection PRIVATE dl)

# ---- Compiler flags ----
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(video_detection PRIVATE -Wall -Wextra -O2 -march=native)
endif()

# ---- Install ----
install(TARGETS video_detection DESTINATION bin)

# ---- Print summary ----
message(STATUS "")
message(STATUS "==============================")
message(STATUS "  VideoDetection Build Config")
message(STATUS "==============================")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "==============================")
