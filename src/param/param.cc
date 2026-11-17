/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Parameter definitions for the ncclParam system itself.

#include "param/param.h"
#include "param/parsers.h"
#include "debug.h"

#include <unordered_set>

DEFINE_NCCL_PARAM(ncclParamDumpAllFlag, bool, NCCL_PARAM_DUMP_ALL, false, NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT,
                  "Print all parameters including private ones");

using ncclStringSet = std::unordered_set<std::string>;
DEFINE_NCCL_PARAM(ncclParamNoCacheStr, const char*, NCCL_NO_CACHE, nullptr, NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT,
                  "Comma-separated list of param keys to disable caching (or ALL)");

// 判断指定参数是否被 NCCL_NO_CACHE 设置为“禁用缓存”
extern "C" bool ncclParamIsCacheDisabled(const char* key) {
  // 指定环境变量不能是 NCCL_NO_CACHE 本身，否则会循环依赖，NCCL_NO_CACHE 永远不禁用缓存
  // Short-circuit for NCCL_NO_CACHE itself to prevent circular dependency
  if (std::strcmp(key, "NCCL_NO_CACHE") == 0) return false;

  static std::once_flag initFlag;
  // std::unordered_set<std::string>
  static ncclStringSet set;
  // 是否所有参数都禁用缓存
  static bool noCacheAll = false;

  std::call_once(initFlag, []() {
    // 创建使用逗号分隔的字符串列表解析器，例如 NCCL_DEBUG,NCCL_DEBUG_FILE,NCCL_NET 会被解析成 {"NCCL_DEBUG", "NCCL_DEBUG_FILE", "NCCL_NET"}
    auto parser = ncclParamListOf<ncclStringSet>(',');
    // 读取 NCCL_NO_CACHE 参数的值并解析到 set 中
    if (parser.resolve(ncclParamNoCacheStr(), set) == ncclSuccess) {
      noCacheAll = set.count("ALL") > 0;
    }
  });

  // 是否禁用缓存
  bool ret = noCacheAll || set.count(key) > 0;
  if (ret) INFO(NCCL_ENV, "PARAM: Disabling caching for environment variable %s.", key);
  return ret;
}

// 该函数负责读取指定 NCCL 参数，决定从以下哪个来源获取：
// 1. EnvPlugin；
// 2. 系统环境变量 std::getenv()。
// Exported helper for ncclParam<T>::loadValue() so plugins can resolve
// a single symbol instead of requiring ncclInitEnv + ncclEnvPluginGetEnv
// to be exported.
#include "env.h"
extern "C" const char* ncclParamEnvPluginGet(const char* key, bool env_init) {
  if (env_init) {
    // regular parameters will init env plugins before reading env
    // 内部有 std::call_once()，即使初始化过了环境插件也无所谓
    ncclInitEnv();
    // 从环境插件链中查找参数值
    return ncclEnvPluginGetEnv(key);
  } else {
    // 不主动初始化；但若插件早已初始化，仍优先使用插件
    // special parameters that do not attempt to initialize env plugins
    return ncclEnvPluginInitialized() ? ncclEnvPluginGetEnv(key) : std::getenv(key);
  }
}
