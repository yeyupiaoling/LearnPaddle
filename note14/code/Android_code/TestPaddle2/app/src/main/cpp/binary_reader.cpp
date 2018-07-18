/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stdlib.h>

#include "binary_reader.h"

#define TAG "BinaryReader"

#ifdef __ANDROID__
#include <android/log.h>
#define LOGI(format, ...) \
  __android_log_print(ANDROID_LOG_INFO, TAG, format, ##__VA_ARGS__)
#define LOGW(format, ...) \
  __android_log_print(ANDROID_LOG_WARN, TAG, format, ##__VA_ARGS__)
#define LOGE(format, ...) \
  __android_log_print(ANDROID_LOG_ERROR, TAG, "Error: " format, ##__VA_ARGS__)
#else
#include <stdio.h>
#define LOGI(format, ...) \
  fprintf(stdout, "[" TAG "]" format "\n", ##__VA_ARGS__)
#define LOGW(format, ...) \
  fprintf(stdout, "[" TAG "]" format "\n", ##__VA_ARGS__)
#define LOGE(format, ...) \
  fprintf(stderr, "[" TAG "]Error: " format "\n", ##__VA_ARGS__)
#endif

void* BinaryReader::read_binary_from_external(const char* filename,
                                              long* size) {
  FILE* file = fopen(filename, "rb");
  if (file == nullptr) {
    LOGE("%s open failure.", filename);
    return nullptr;
  }

  fseek(file, 0L, SEEK_END);
  *size = ftell(file);
  fseek(file, 0L, SEEK_SET);

  void* buf = malloc(*size);
  if (buf == nullptr) {
    LOGE("memory allocation failure, size %ld.", *size);
    return nullptr;
  }

  fread(buf, 1, *size, file);

  fclose(file);
  file = nullptr;

  return buf;
}

#ifdef __ANDROID__
AAssetManager* BinaryReader::aasset_manager_(nullptr);

void* BinaryReader::read_binary_from_asset(const char* filename, long* size) {
  if (aasset_manager_ != nullptr) {
    AAsset* asset =
        AAssetManager_open(aasset_manager_, filename, AASSET_MODE_STREAMING);

    if (asset != nullptr) {
      *size = AAsset_getLength(asset);

      void* buf = (char*)malloc(*size);
      if (buf == nullptr) {
        LOGW("memory allocation failure, size %ld", *size);
        return nullptr;
      }

      if (AAsset_read(asset, buf, *size) > 0) {
        AAsset_close(asset);
        return buf;
      } else {
        LOGW("read %s failure, size %ld.", filename, *size);
      }

      AAsset_close(asset);
      asset = nullptr;
    } else {
      LOGW("%s does not exist in assets.", filename);
    }
  }

  return nullptr;
}
#endif
