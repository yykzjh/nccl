/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdlib.h>
#include "debug.h"
#include "nccl_env.h"

// 默认环境插件初始化函数
static ncclResult_t ncclEnvInit(uint8_t ncclMajor, uint8_t ncclMinor, uint8_t ncclPatch, const char* suffix,
                                ncclDebugLogger_t logFunction) {
  return ncclSuccess;
}

// 默认环境插件销毁函数
static ncclResult_t ncclEnvFinalize(void) {
  return ncclSuccess;
}

// 默认环境插件获取环境变量函数
static const char* ncclEnvGetEnv(const char* name) {
  return std::getenv(name);
}

// 默认环境插件结构体
ncclEnv_v2_t ncclIntEnv_v2 = {
  "ncclEnvDefault",
  ncclEnvInit,
  ncclEnvFinalize,
  ncclEnvGetEnv,
};
