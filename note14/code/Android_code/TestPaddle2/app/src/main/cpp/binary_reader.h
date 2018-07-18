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

#pragma once

#ifdef __ANDROID__
#include <android/asset_manager.h>
#endif

class BinaryReader {
public:
  void* operator()(const char* filename, long* size) {
#ifdef __ANDROID__
    void* buf = read_binary_from_asset(filename, size);
#else
    void* buf = NULL;
#endif
    if (buf == NULL) {
      buf = read_binary_from_external(filename, size);
    }
    return buf;
  }

private:
  void* read_binary_from_external(const char* filename, long* size);

#ifdef __ANDROID__
private:
  void* read_binary_from_asset(const char* filename, long* size);

public:
  static void set_aasset_manager(AAssetManager* aasset_manager) {
    aasset_manager_ = aasset_manager;
  }

private:
  static AAssetManager* aasset_manager_;
#endif
};
