cmake_minimum_required(VERSION 3.4.1)

project(aibox)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread -g")

set(EXECUTABLE_NAME "aibox")

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# rknn api
# set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/lib/librknnrt.so)

#rga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588)
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)

# opencv
find_package(OpenCV REQUIRED)

# module
set(MODULE_PATH ${CMAKE_SOURCE_DIR}/model)

# include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/3rdparty)
include_directories(${RGA_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/sort/include)

# 添加你的新类
add_executable(${EXECUTABLE_NAME}
        src/main.cpp
        src/FalldownDetect.cpp
        src/FireSmokeDetect.cpp
        src/PersonAttribute.cpp
        src/PersonDetect.cpp
        src/postprocess.cpp
        src/preprocess.cpp
        src/FileUtils.c
        sort/src/Hungarian.cc
        sort/src/KalmanTracker.cc
        sort/src/sort.cc
)

# 链接库
target_link_libraries(aibox
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
  ${RGA_LIB}
)

# install target and libraries
install(TARGETS ${EXECUTABLE_NAME} DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(DIRECTORY ${MODULE_PATH} DESTINATION ./)
