#include <jni.h>
#include <paddle/capi.h>
#include <string.h>
#include <vector>
#include "binary_reader.h"

#define TAG "PaddlePaddle"

#ifdef __ANDROID__

#include <android/log.h>
#include <stdlib.h>

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

static const char* paddle_error_string(paddle_error status) {
  switch (status) {
    case kPD_NULLPTR:
      return "nullptr error";
    case kPD_OUT_OF_RANGE:
      return "out of range error";
    case kPD_PROTOBUF_ERROR:
      return "protobuf error";
    case kPD_NOT_SUPPORTED:
      return "not supported error";
    case kPD_UNDEFINED_ERROR:
      return "undefined error";
    default:
      return "";
  };
}

#define CHECK(stmt)                                   \
  do {                                                \
    paddle_error __err__ = stmt;                      \
    if (__err__ != kPD_NO_ERROR) {                    \
      const char* str = paddle_error_string(__err__); \
      LOGE("%s (%d) in " #stmt "\n", str, __err__);   \
      exit(__err__);                                  \
    }                                                 \
  } while (0)