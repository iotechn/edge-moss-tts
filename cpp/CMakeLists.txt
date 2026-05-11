cmake_minimum_required(VERSION 3.16)
project(moss_tts_onnx_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ONNXRUNTIME REQUIRED IMPORTED_TARGET libonnxruntime)

find_path(SENTENCEPIECE_INCLUDE sentencepiece_processor.h
  PATHS /opt/homebrew/opt/sentencepiece/include /usr/local/include /usr/include)
find_library(SENTENCEPIECE_LIB NAMES sentencepiece
  PATHS /opt/homebrew/opt/sentencepiece/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
if(NOT SENTENCEPIECE_INCLUDE OR NOT SENTENCEPIECE_LIB)
  message(FATAL_ERROR "sentencepiece not found; install via: brew install sentencepiece")
endif()

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
  cpphttplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.3
)
FetchContent_MakeAvailable(cpphttplib)

add_executable(moss_tts_onnx_server
  src/main.cpp
  src/engine.cpp
)

target_include_directories(moss_tts_onnx_server PRIVATE
  ${SENTENCEPIECE_INCLUDE}
  ${cpphttplib_SOURCE_DIR}
)

target_link_libraries(moss_tts_onnx_server PRIVATE
  PkgConfig::ONNXRUNTIME
  ${SENTENCEPIECE_LIB}
  nlohmann_json::nlohmann_json
)

if(APPLE)
  target_link_libraries(moss_tts_onnx_server PRIVATE "-framework Foundation")
endif()
