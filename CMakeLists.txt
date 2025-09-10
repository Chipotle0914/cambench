cmake_minimum_required(VERSION 3.16)
project(cambenchpp VERSION 0.1.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(CAM_ENABLE_OPENCV "Build with OpenCV support" ON)

if (CAM_ENABLE_OPENCV)
  find_package(OpenCV REQUIRED)
endif()
find_package(yaml-cpp REQUIRED)   


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


add_executable(cambench
  src/main.cpp
)


if (CAM_ENABLE_OPENCV)
  target_link_libraries(cambench PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(cambench PRIVATE CAM_HAVE_OPENCV=1)
endif()
target_link_libraries(cambench PRIVATE yaml-cpp)


if (MSVC)
  target_compile_options(cambench PRIVATE /W4 /permissive-)
else()
  target_compile_options(cambench PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS cambench RUNTIME DESTINATION bin)
