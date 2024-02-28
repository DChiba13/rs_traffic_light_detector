cmake_minimum_required(VERSION 3.0)
project(rs_traffic_light_detector VERSION 1.1.0 LANGUAGES CXX)

# ビルドフラグの設定
set(CMAKE_CXX_STANDARD 17)

# OpenCVの検出と設定
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# インクルードパスの追加
include_directories(/usr/local/include)

# ビルドに関する設定
set(EXECUTABLES "rs_traffic_light_detector")
set(DEPENDENT_LIBS ${OpenCV_LIBRARIES} rs_common)

foreach(target IN LISTS EXECUTABLES)
  add_executable(${target} src/${target}.cpp)
  target_link_libraries(${target} ${DEPENDENT_LIBS} stdc++fs)
endforeach()
