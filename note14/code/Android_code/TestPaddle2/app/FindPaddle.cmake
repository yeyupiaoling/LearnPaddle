# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

if(NOT PADDLE_ROOT)
   set(PADDLE_ROOT $ENV{PADDLE_ROOT} CACHE PATH "Paddle Path")
endif()
if(NOT PADDLE_ROOT)
  message(FATAL_ERROR "Set PADDLE_ROOT as your root directory installed PaddlePaddle")
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)

find_path(PADDLE_INC_DIR NAMES paddle/capi.h PATHS ${PADDLE_ROOT})
message(TEST_PATH "测试目录:${PADDLE_ROOT}")
find_library(PADDLE_SHARED_LIB NAMES paddle_capi_shared PATHS
    ${PADDLE_ROOT}/lib/${ANDROID_ABI})
message(TEST_PATH_1 "测试目录:${ANDROID_ABI}")
find_library(PADDLE_WHOLE_LIB NAMES paddle_capi_whole PATHS
    ${PADDLE_ROOT}/lib/${ANDROID_ABI})
find_library(PADDLE_LAYERS_LIB NAMES paddle_capi_layers PATHS
    ${PADDLE_ROOT}/lib/${ANDROID_ABI})
find_library(PADDLE_ENGINE_LIB NAMES paddle_capi_engine PATHS
    ${PADDLE_ROOT}/lib/${ANDROID_ABI})
if(PADDLE_INC_DIR AND PADDLE_LAYERS_LIB AND PADDLE_ENGINE_LIB)
  add_library(paddle_capi_layers STATIC IMPORTED)
  set_target_properties(paddle_capi_layers PROPERTIES IMPORTED_LOCATION
                        ${PADDLE_LAYERS_LIB})
  add_library(paddle_capi_engine STATIC IMPORTED)
  set_target_properties(paddle_capi_engine PROPERTIES IMPORTED_LOCATION
                        ${PADDLE_ENGINE_LIB})
  set(PADDLE_LIBRARIES -Wl,--start-group
                       -Wl,--whole-archive paddle_capi_layers
                       -Wl,--no-whole-archive paddle_capi_engine
                       -Wl,--end-group)
  message(STATUS "Found PaddlePaddle (include: ${PADDLE_INC_DIR}; "
                 "library: ${PADDLE_LAYERS_LIB}, ${PADDLE_ENGINE_LIB})")
elseif(PADDLE_INC_DIR AND PADDLE_WHOLE_LIB)
  add_library(paddle_capi_whole STATIC IMPORTED)
  set_target_properties(paddle_capi_whole PROPERTIES IMPORTED_LOCATION
                        ${PADDLE_WHOLE_LIB})
  set(PADDLE_LIBRARIES -Wl,--whole-archive paddle_capi_whole -Wl,--no-whole-archive)
else()
  message(FATAL_ERROR "Cannot find PaddlePaddle on ${PADDLE_ROOT}\n"
          "\tPADDLE_INC_DIR: ${PADDLE_INC_DIR}\n"
          "\tPADDLE_LAYERS_LIB: ${PADDLE_LAYERS_LIB}\n"
          "\tPADDLE_ENGINE_LIB: ${PADDLE_ENGINE_LIB}\n"
          "\tPADDLE_WHOLE_LIB: ${PADDLE_WHOLE_LIB}")
endif()

include_directories(${PADDLE_INC_DIR})

set(THIRD_PARTY_ROOT ${PADDLE_ROOT}/third_party)
function(third_party_library TARGET_NAME HEADER_NAME LIBRARY_NAME)
  find_path(${TARGET_NAME}_INC_DIR ${HEADER_NAME} PATHS
      ${THIRD_PARTY_ROOT}/${TARGET_NAME}/include
      NO_DEFAULT_PATH)
  find_library(${TARGET_NAME}_STATIC_LIBRARY NAMES ${LIBRARY_NAME} PATHS
      ${THIRD_PARTY_ROOT}/${TARGET_NAME}/lib/${ANDROID_ABI}
      NO_DEFAULT_PATH)
  if(${TARGET_NAME}_INC_DIR AND ${TARGET_NAME}_STATIC_LIBRARY)
    add_library(${TARGET_NAME} STATIC IMPORTED)
    set_target_properties(${TARGET_NAME} PROPERTIES IMPORTED_LOCATION
        ${${TARGET_NAME}_STATIC_LIBRARY})
    set(PADDLE_THIRD_PARTY_LIBRARIES
        ${PADDLE_THIRD_PARTY_LIBRARIES} ${TARGET_NAME} PARENT_SCOPE)
    message(STATUS "Found ${TARGET_NAME}: " ${${TARGET_NAME}_STATIC_LIBRARY})
  else()
    message(WARNING "Cannot find ${TARGET_NAME} under ${THIRD_PARTY_ROOT}")
  endif()
endfunction()

set(PADDLE_THIRD_PARTY_LIBRARIES)
third_party_library(protobuf google/protobuf/message.h protobuf)
third_party_library(glog glog/logging.h glog)
third_party_library(openblas cblas.h openblas)
third_party_library(gflags gflags/gflags.h gflags)
third_party_library(zlib zlib.h z)

set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)

